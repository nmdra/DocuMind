from __future__ import annotations

import argparse
import logging
import uuid
from collections.abc import Mapping
from typing import Any

import chromadb
import ollama as ollama_client
from fastmcp import Context, FastMCP
from fastmcp.utilities.logging import get_logger

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DEFAULT_LOG_LEVEL,
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
logger = logging.getLogger(__name__)
to_client_logger = get_logger(name="fastmcp.server.context.to_client")


def _configure_logging(level_name: str, to_client_debug: bool) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if to_client_debug:
        to_client_logger.setLevel(logging.DEBUG)


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
async def add_document(
    text: str,
    doc_id: str | None = None,
    source: str = "",
    ctx: Context | None = None,
) -> str:
    """Embed and store a document in ChromaDB.

    Args:
        text: The document text to store.
        doc_id: Optional unique identifier (auto-generated if omitted).
        source: Optional source label (filename, URL, etc.).
    """
    _id = doc_id or str(uuid.uuid4())
    if ctx:
        await ctx.info(
            f"Storing document {_id}",
            extra={"doc_id": _id, "source": source, "text_length": len(text)},
        )
    vec = _embed(text)

    try:
        col.upsert(
            ids=[_id],
            embeddings=[vec],
            documents=[text],
            metadatas=[{"source": source, "length": len(text)}],
        )
    except Exception as exc:  # pragma: no cover - external DB failure path
        if ctx:
            await ctx.error(
                "Failed to upsert document into ChromaDB",
                extra={"doc_id": _id, "source": source, "error": str(exc)},
            )
        raise RuntimeError("Failed to upsert document into ChromaDB.") from exc

    if ctx:
        await ctx.debug("Document stored", extra={"doc_id": _id})
    return f"Stored document {_id} ({len(text)} chars, source={source!r})"


@mcp.tool()
async def semantic_search(
    query: str,
    n_results: int = TOP_K,
    source_filter: str = "",
    ctx: Context | None = None,
) -> str:
    """Search ChromaDB for documents semantically similar to the query.

    Args:
        query: Natural-language search query.
        n_results: Number of results to return.
        source_filter: If set, restrict results to this source label.
    """
    if n_results < 1:
        if ctx:
            await ctx.warning("Invalid n_results provided", extra={"n_results": n_results})
        raise ValueError("n_results must be >= 1")

    if ctx:
        await ctx.info(
            "Running semantic search",
            extra={
                "query_length": len(query),
                "n_results": n_results,
                "source_filter": source_filter,
            },
        )
    vec = _embed(query)
    where = {"source": source_filter} if source_filter else None

    try:
        results = col.query(
            query_embeddings=[vec],
            n_results=n_results,
            where=where,
        )
    except Exception as exc:  # pragma: no cover - external DB failure path
        if ctx:
            await ctx.error(
                "Failed to query ChromaDB",
                extra={"n_results": n_results, "source_filter": source_filter, "error": str(exc)},
            )
        raise RuntimeError("Failed to query ChromaDB.") from exc

    docs = (results.get("documents") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]

    if not docs:
        if ctx:
            await ctx.warning("Semantic search returned no results")
        return "No results found."

    lines: list[str] = []
    for i, (doc, dist, meta) in enumerate(zip(docs, dists, metas, strict=True), start=1):
        metadata = _as_metadata(meta)
        score = 1.0 - dist if isinstance(dist, (float, int)) else 0.0
        lines.append(f"[{i}] score={score:.3f} source={metadata.get('source', '')}")
        lines.append((doc or "")[:MAX_RESULT_PREVIEW_LENGTH])
        lines.append("")

    if ctx:
        await ctx.debug("Semantic search completed", extra={"result_count": len(docs)})
    return "\n".join(lines).strip() if lines else "No results found."


@mcp.tool()
async def collection_stats(ctx: Context | None = None) -> str:
    """Return document count and collection metadata."""
    try:
        count = col.count()
    except Exception as exc:  # pragma: no cover - external DB failure path
        if ctx:
            await ctx.error("Failed to read collection statistics", extra={"error": str(exc)})
        raise RuntimeError("Failed to read collection statistics from ChromaDB.") from exc

    if ctx:
        await ctx.info("Read collection statistics", extra={"collection": COLLECTION_NAME, "count": count})
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
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help="Python logging level (e.g., DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--to-client-debug",
        action="store_true",
        help="Enable DEBUG logs on fastmcp.server.context.to_client logger.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _configure_logging(args.log_level, args.to_client_debug)
    logger.info("Starting server transport=%s", args.transport)
    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
