# DocuMind

Fully local RAG agent built with FastMCP + ChromaDB + Ollama models (`phi4-mini:3.8b-q4_K_M` and `embeddinggemma:300m-qat-q8_0`), managed with UV and linted/formatted with Ruff.

## Privacy and locality

All components run on your machine:
- LLM inference via local Ollama
- Embedding generation via local Ollama
- Vector storage/query via local ChromaDB

No cloud APIs are required.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- [UV](https://docs.astral.sh/uv/) installed

Install/pull models:

```bash
ollama pull phi4-mini:3.8b-q4_K_M
ollama pull embeddinggemma:300m-qat-q8_0
ollama list
```

## Project layout

```text
/home/runner/work/DocuMind/DocuMind/
├── pyproject.toml
├── uv.lock
├── .python-version
├── config.py
├── server.py
├── client.py
├── ingest.py
├── data/
└── chroma_db/   # runtime-created, ignored by git
```

## Setup

```bash
cd /home/runner/work/DocuMind/DocuMind
python3 -m uv sync
```

## Ingest documents

```bash
python3 -m uv run python ingest.py /home/runner/work/DocuMind/DocuMind/data/my_notes.txt
python3 -m uv run python ingest.py /home/runner/work/DocuMind/DocuMind/data/notes.txt /home/runner/work/DocuMind/DocuMind/data/report.md
```

## Run the agent

Start Ollama if needed:

```bash
ollama serve
```

Launch interactive client:

```bash
cd /home/runner/work/DocuMind/DocuMind
python3 -m uv run python client.py
```

## FastMCP tools (MVP)

- `add_document(text, doc_id=None, source="")`
- `semantic_search(query, n_results=5, source_filter="")`
- `collection_stats()`

## Ruff workflow

```bash
cd /home/runner/work/DocuMind/DocuMind
python3 -m uv run ruff format .
python3 -m uv run ruff check --fix .
python3 -m uv run ruff format --check . && python3 -m uv run ruff check .
```

## Operations checks

Verify collection count:

```bash
cd /home/runner/work/DocuMind/DocuMind
python3 -m uv run python -c "import chromadb; c=chromadb.PersistentClient('./chroma_db'); print(c.get_or_create_collection('documents').count())"
```

## Troubleshooting

- `Connection refused` on `localhost:11434`
  - Ensure `ollama serve` is running.
- Missing model errors
  - Re-run `ollama pull phi4-mini:3.8b-q4_K_M` and `ollama pull embeddinggemma:300m-qat-q8_0`.
- Empty search results
  - Check ingestion completed and collection count is non-zero.
- ChromaDB embedding dimension mismatch
  - Keep one embedding model per collection; clear `chroma_db/` and re-ingest if model changes.
