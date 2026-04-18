from __future__ import annotations

import argparse
import asyncio
import logging
import re
import shlex
from collections.abc import Mapping

import chromadb
import fastmcp.client.logging as fastmcp_logging
import ollama
from fastmcp import Client
from rich.console import Console
from rich.markdown import Markdown

from config import (
    CHAT_MODEL,
    CHROMA_PATH,
    CONVERSATION_COLLECTION_NAME,
    DEFAULT_LOG_LEVEL,
    DEFAULT_CONVERSATION_SESSION_ID,
    DEFAULT_SERVER_COMMAND,
    DEFAULT_TRANSPORT,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    SSE_HOST,
    SSE_PORT,
    TOP_K,
)

console = Console()
logger = logging.getLogger(__name__)
ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
chroma = chromadb.PersistentClient(path=CHROMA_PATH)
conversation_col = chroma.get_or_create_collection(
    name=CONVERSATION_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
EMPTY_MESSAGE_SENTINEL = "__EMPTY_MESSAGE__"
LOGGING_LEVEL_MAP = logging.getLevelNamesMapping()
STRICT_RAG_FALLBACK = "I cannot answer this based on the provided documents."
STRICT_RAG_SYSTEM_PROMPT = (
    "You are a strict, factual assistant. You will be provided with context from specific documents.\n\n"
    "Your rules:\n"
    "1. You must answer the user's question ONLY using the information provided in the context blocks.\n"
    "2. Do not use your pre-trained knowledge or outside information.\n"
    f"3. If the context does not contain the answer, output exactly: '{STRICT_RAG_FALLBACK}'\n"
    "4. If you can answer, every sentence must include a citation in the form [Source: <source>].\n"
    "5. Cite only sources that appear in the provided context blocks."
)
SOURCE_CITATION_PATTERN = re.compile(r"\[Source:\s*([^\]]+)\]")
SOURCE_LINE_PATTERN = re.compile(r"source=(.+?)\s*$")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
NO_RESULTS_PREFIX = "no results found"


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


async def _server_log_handler(message: fastmcp_logging.LogMessage) -> None:
    """Handle MCP server log messages by forwarding them to Python logging."""
    data = message.data if isinstance(message.data, Mapping) else {}
    msg = data.get("msg")
    if not isinstance(msg, str) or not msg:
        msg = str(data) if data else repr(message)
    if isinstance(message.level, str):
        level = LOGGING_LEVEL_MAP.get(message.level.upper(), logging.INFO)
    elif isinstance(message.level, int) and logging.DEBUG <= message.level <= logging.CRITICAL:
        level = message.level
    else:
        level = logging.INFO
    logger.log(level, "[mcp:%s] %s", message.logger or "server", msg)


def _tool_result_text(result: object) -> str:
    """Extract text content from a tool result object or dict payload."""
    content = getattr(result, "content", None)
    if not isinstance(content, list) and isinstance(result, Mapping):
        content = result.get("content")
    if not isinstance(content, list):
        return ""
    first = next(iter(content), None)
    if first is None:
        return ""
    text = getattr(first, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(first, Mapping):
        text = first.get("text")
        if isinstance(text, str):
            return text
    return str(first)


def _extract_sources_from_context(context_text: str) -> set[str]:
    sources: set[str] = set()
    for line in context_text.splitlines():
        match = SOURCE_LINE_PATTERN.search(line.strip())
        if not match:
            continue
        source = match.group(1).strip()
        if source:
            sources.add(source)
    return sources


def _is_refusal(content: str) -> bool:
    return content.strip() == STRICT_RAG_FALLBACK


def _has_sentence_level_citations(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return False
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(stripped) if sentence.strip()]
    if not sentences:
        return False
    return all(SOURCE_CITATION_PATTERN.search(sentence) for sentence in sentences)


def _citations_match_sources(content: str, allowed_sources: set[str]) -> bool:
    if not allowed_sources:
        return False
    cited_sources: set[str] = set()
    for match in SOURCE_CITATION_PATTERN.findall(content):
        source = match.strip()
        if source:
            cited_sources.add(source)
    if not cited_sources:
        return False
    return cited_sources.issubset(allowed_sources)


def _answer_is_valid(content: str, allowed_sources: set[str]) -> bool:
    if _is_refusal(content):
        return True
    return _has_sentence_level_citations(content) and _citations_match_sources(content, allowed_sources)


def _context_block(context_text: str) -> str:
    return f"Context blocks:\n{context_text}"


def _is_no_results_context(context_text: str) -> bool:
    normalized = context_text.strip().lower()
    if not normalized:
        return True
    return normalized.startswith(NO_RESULTS_PREFIX)


def _embed(text: str) -> list[float]:
    try:
        resp = ollama_client.embed(model=EMBED_MODEL, input=text)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate embedding with Ollama model {EMBED_MODEL!r}."
        ) from exc
    embeddings = resp.get("embeddings")
    embedding = embeddings[0] if isinstance(embeddings, list) and embeddings else None
    if not isinstance(embedding, list):
        raise RuntimeError("Embedding response from Ollama was malformed.")
    return embedding


def _load_history(session_id: str) -> list[dict]:
    results = conversation_col.get(
        where={"session_id": session_id}, include=["documents", "metadatas"]
    )
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []

    items: list[tuple[int, dict]] = []
    for doc, meta in zip(docs, metas, strict=True):
        if not isinstance(meta, dict):
            continue
        if meta.get("event") == "tool_calls":
            continue
        role = meta.get("role")
        turn = meta.get("turn")
        if not isinstance(role, str) or not isinstance(turn, int):
            continue

        msg: dict[str, str] = {"role": role, "content": doc or ""}
        name = meta.get("name")
        if isinstance(name, str) and name:
            msg["name"] = name

        items.append((turn, msg))

    items.sort(key=lambda item: item[0])
    return [msg for _, msg in items]


def _persist_message(
    session_id: str,
    turn: int,
    message: dict,
    extra_metadata: Mapping[str, int | str] | None = None,
) -> None:
    role = message.get("role", "")
    content = message.get("content", "")
    if not isinstance(role, str) or not isinstance(content, str):
        return

    metadata: dict[str, int | str] = {
        "session_id": session_id,
        "turn": turn,
        "role": role,
    }
    name = message.get("name")
    if isinstance(name, str) and name:
        metadata["name"] = name
    if extra_metadata:
        metadata.update(extra_metadata)

    conversation_col.upsert(
        ids=[f"{session_id}-{turn:08d}"],
        embeddings=[_embed(content or EMPTY_MESSAGE_SENTINEL)],
        documents=[content],
        metadatas=[metadata],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive client for the local FastMCP server.")
    parser.add_argument(
        "--session-id",
        default=DEFAULT_CONVERSATION_SESSION_ID,
        help="Conversation session ID for persisted memory.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default=DEFAULT_TRANSPORT,
        help="Transport used to connect to the MCP server.",
    )
    parser.add_argument(
        "--server-command",
        default=DEFAULT_SERVER_COMMAND,
        help="Command used to start the MCP server subprocess when transport=stdio.",
    )
    parser.add_argument(
        "--sse-url",
        default=f"http://{SSE_HOST}:{SSE_PORT}/sse",
        help="SSE endpoint URL when transport=sse.",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help="Python logging level (e.g., DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


async def run_agent(
    session_id: str,
    server_command: str,
    transport: str,
    sse_url: str,
) -> None:
    if transport == "stdio":
        parts = shlex.split(server_command)
        if not parts:
            raise ValueError("Server command cannot be empty.")
        server_connection = server_command
    elif transport == "sse":
        server_connection = sse_url
    else:
        raise ValueError(f"Unsupported transport: {transport!r}. Expected one of: 'stdio', 'sse'.")

    async with Client(server_connection, log_handler=_server_log_handler) as session:
        try:
            tool_resp = await session.list_tools()
            raw_tools = getattr(tool_resp, "tools", tool_resp)
            tool_names = set()
            for t in (raw_tools if isinstance(raw_tools, list) else []):
                if isinstance(t, Mapping):
                    name = t.get("name")
                else:
                    name = getattr(t, "name", None)
                if isinstance(name, str) and name:
                    tool_names.add(name)
        except Exception:  # pragma: no cover - transport/tool-list failure path
            logger.exception("Failed to list server tools. Agent will continue without server tools.")
            console.print(
                "[bold yellow]Warning:[/bold yellow] Failed to list MCP tools; continuing without tool access."
            )
            tool_names = set()
        has_semantic_search = "semantic_search" in tool_names
        if not has_semantic_search:
            console.print(
                "[bold yellow]Warning:[/bold yellow] semantic_search tool is unavailable; responses will fallback."
            )

        console.print("[bold green]FastMCP + ChromaDB agent ready.[/bold green]")
        console.print(f"[dim]{CHAT_MODEL} | {EMBED_MODEL}[/dim]")
        console.print(f"[dim]Session: {session_id} | transport={transport}[/dim]")
        console.print("[dim]Type 'quit' to exit.[/dim]")

        def _next_persisted_turn(session_id: str) -> int:
            try:
                client = chromadb.PersistentClient(path=CHROMA_PATH)
                collection = client.get_or_create_collection(name=CONVERSATION_COLLECTION_NAME)
                records = collection.get(
                    where={"session_id": session_id},
                    include=["metadatas"],
                )
            except Exception:
                return 0

            max_turn = -1
            for metadata in records.get("metadatas") or []:
                if isinstance(metadata, Mapping):
                    stored_turn = metadata.get("turn")
                    if isinstance(stored_turn, int):
                        max_turn = max(max_turn, stored_turn)

            if max_turn >= 0:
                return max_turn + 1

            for record_id in records.get("ids") or []:
                if not isinstance(record_id, str):
                    continue
                prefix, sep, suffix = record_id.rpartition("-")
                if prefix == session_id and sep and suffix.isdigit():
                    max_turn = max(max_turn, int(suffix))

            return max_turn + 1 if max_turn >= 0 else 0

        history = _load_history(session_id)
        turn = _next_persisted_turn(session_id)
        if history:
            console.print(f"[dim]Loaded {len(history)} persisted message(s).[/dim]")

        while True:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                break

            user_message = {"role": "user", "content": user_input}
            history.append(user_message)
            _persist_message(session_id, turn, user_message)
            turn += 1

            if not has_semantic_search:
                assistant_text = STRICT_RAG_FALLBACK
            else:
                context_result = await session.call_tool(
                    "semantic_search",
                    {"query": user_input, "n_results": TOP_K},
                )
                context_text = _tool_result_text(context_result)
                if _is_no_results_context(context_text):
                    assistant_text = STRICT_RAG_FALLBACK
                else:
                    allowed_sources = _extract_sources_from_context(context_text)
                    request_messages = [
                        {"role": "system", "content": STRICT_RAG_SYSTEM_PROMPT},
                        *history,
                        {"role": "system", "content": _context_block(context_text)},
                    ]
                    try:
                        response = ollama_client.chat(model=CHAT_MODEL, messages=request_messages)
                    except Exception as exc:  # pragma: no cover - network/service failure path
                        console.print(f"[bold red]Ollama request failed:[/bold red] {exc}")
                        continue
                    content = response.message.content or ""
                    assistant_text = (
                        content if _answer_is_valid(content, allowed_sources) else STRICT_RAG_FALLBACK
                    )

            console.print("[bold green]Agent:[/bold green]")
            console.print(Markdown(assistant_text))
            assistant_message = {"role": "assistant", "content": assistant_text}
            history.append(assistant_message)
            _persist_message(session_id, turn, assistant_message)
            turn += 1


if __name__ == "__main__":
    args = parse_args()
    _configure_logging(args.log_level)
    asyncio.run(
        run_agent(
            args.session_id,
            args.server_command,
            args.transport,
            args.sse_url,
        )
    )
