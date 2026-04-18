from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
import ollama as ollama_client

from config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    SUPPORTED_INGEST_EXTENSIONS,
)

chroma = chromadb.PersistentClient(path=CHROMA_PATH)
col = chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
ollama = ollama_client.Client(host=OLLAMA_BASE_URL)


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")

    step = CHUNK_SIZE - CHUNK_OVERLAP
    if step <= 0:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step

    return chunks


def embed_text(text: str) -> list[float]:
    """Embed text using the configured Ollama embedding model."""
    try:
        resp = ollama.embed(model=EMBED_MODEL, input=text)
    except Exception as exc:  # pragma: no cover - external service failure path
        raise RuntimeError(
            f"Failed to embed text with model '{EMBED_MODEL}'. "
            "Ensure Ollama is running and the model is pulled."
        ) from exc

    embeddings = resp.get("embeddings")
    embedding = embeddings[0] if isinstance(embeddings, list) and embeddings else None
    if not isinstance(embedding, list):
        raise RuntimeError("Embedding response from Ollama was malformed.")

    return embedding


def ingest_file(path: Path) -> None:
    """Ingest a single UTF-8 file into ChromaDB."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_INGEST_EXTENSIONS:
        supported = ", ".join(SUPPORTED_INGEST_EXTENSIONS)
        raise ValueError(f"Unsupported file type for v1 ingestion: {path.suffix!r}. Use: {supported}")

    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    print(f"Ingesting {path.name}: {len(chunks)} chunk(s)")

    for i, chunk in enumerate(chunks):
        doc_id = f"{path.stem}-{i}"
        col.upsert(
            ids=[doc_id],
            embeddings=[embed_text(chunk)],
            documents=[chunk],
            metadatas=[{"source": path.name, "chunk": i}],
        )
        print(f"  [{i + 1}/{len(chunks)}] {doc_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest text/markdown files into the local ChromaDB collection."
    )
    parser.add_argument("files", nargs="+", help="One or more .txt/.md/.markdown files to ingest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for file_arg in args.files:
        ingest_file(Path(file_arg))

    print(f"Done. Collection size: {col.count()} chunk(s)")


if __name__ == "__main__":
    main()
