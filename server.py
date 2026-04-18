from __future__ import annotations

import argparse
import uuid
from collections.abc import Mapping
from typing import Any

import chromadb
import ollama as ollama_client
from fastmcp import FastMCP

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DEFAULT_TRANSPORT,
    EMBED_MODEL,
    MAX_RESULT_PREVIEW_LENGTH,
    OLLAMA_BASE_URL,
    SSE_HOST,
    SSE_PORT,
    TOP_K,
)

chroma = chromadb.PersistentClient(path=CHROMA_PATH)
col = chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

mcp = FastMCP("chromadb-tools")
ollama = ollama_client.Client(host=OLLAMA_BASE_URL)


def _embed(text: str) -> list[float]:
    """Embed text with the configured Ollama embedding model."""
    try:
        resp = ollama.embed(model=EMBED_MODEL, input=text)
    except Exception as exc:  # pragma: no cover - external service failure path
        raise RuntimeError(
            f"Failed to generate embeddings using model '{EMBED_MODEL}'. "
            "Ensure Ollama is running and the model is pulled."
        ) from exc

    embeddings = resp.get("embeddings")
    embedding = embeddings[0] if isinstance(embeddings, list) and embeddings else None
    if not isinstance(embedding, list):
        raise RuntimeError("Embedding response from Ollama was malformed.")

    return embedding


def _as_metadata(meta: Any) -> Mapping[str, Any]:
    if isinstance(meta, Mapping):
        return meta
    return {}


@mcp.tool()
def add_document(text: str, doc_id: str | None = None, source: str = "") -> str:
    """Embed and store a document in ChromaDB.

    Args:
        text: The document text to store.
        doc_id: Optional unique identifier (auto-generated if omitted).
        source: Optional source label (filename, URL, etc.).
    """
    _id = doc_id or str(uuid.uuid4())
    vec = _embed(text)

    try:
        col.upsert(
            ids=[_id],
            embeddings=[vec],
            documents=[text],
            metadatas=[{"source": source, "length": len(text)}],
        )
    except Exception as exc:  # pragma: no cover - external DB failure path
        raise RuntimeError("Failed to upsert document into ChromaDB.") from exc

    return f"Stored document {_id} ({len(text)} chars, source={source!r})"


@mcp.tool()
def semantic_search(query: str, n_results: int = TOP_K, source_filter: str = "") -> str:
    """Search ChromaDB for documents semantically similar to the query.

    Args:
        query: Natural-language search query.
        n_results: Number of results to return.
        source_filter: If set, restrict results to this source label.
    """
    if n_results < 1:
        return "n_results must be >= 1"

    vec = _embed(query)
    where = {"source": source_filter} if source_filter else None

    try:
        results = col.query(
            query_embeddings=[vec],
            n_results=n_results,
            where=where,
        )
    except Exception as exc:  # pragma: no cover - external DB failure path
        raise RuntimeError("Failed to query ChromaDB.") from exc

    docs = (results.get("documents") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]

    if not docs:
        return "No results found."

    lines: list[str] = []
    for i, (doc, dist, meta) in enumerate(zip(docs, dists, metas, strict=True), start=1):
        metadata = _as_metadata(meta)
        score = 1.0 - dist if isinstance(dist, (float, int)) else 0.0
        lines.append(f"[{i}] score={score:.3f} source={metadata.get('source', '')}")
        lines.append((doc or "")[:MAX_RESULT_PREVIEW_LENGTH])
        lines.append("")

    return "\n".join(lines).strip() if lines else "No results found."


@mcp.tool()
def collection_stats() -> str:
    """Return document count and collection metadata."""
    try:
        count = col.count()
    except Exception as exc:  # pragma: no cover - external DB failure path
        raise RuntimeError("Failed to read collection statistics from ChromaDB.") from exc

    return f"Collection '{COLLECTION_NAME}': {count} document chunk(s) stored."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastMCP Chroma tools server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default=DEFAULT_TRANSPORT,
        help="Transport mode to run the server with (default: stdio).",
    )
    parser.add_argument("--host", default=SSE_HOST, help="Host for SSE transport.")
    parser.add_argument("--port", type=int, default=SSE_PORT, help="Port for SSE transport.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
