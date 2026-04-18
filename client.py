from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shlex
import time
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
    DEFAULT_CONVERSATION_SESSION_ID,
    DEFAULT_LOG_LEVEL,
    DEFAULT_SERVER_COMMAND,
    DEFAULT_TRANSPORT,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    SSE_HOST,
    SSE_PORT,
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
# Prevents runaway LLM/tool loops from spinning forever in a single user turn.
MAX_TOOL_ITERATIONS_PER_TURN = 8


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


def _tool_defs(tools) -> list[dict]:
    """Convert MCP tools into Ollama function-tool definitions."""
    defs: list[dict] = []
    for t in tools:
        name = getattr(t, "name", None)
        if not isinstance(name, str) and isinstance(t, Mapping):
            name = t.get("name")
        if not isinstance(name, str) or not name:
            continue

        description = getattr(t, "description", "")
        if not isinstance(description, str) and isinstance(t, Mapping):
            description = t.get("description", "")

        schema = getattr(t, "inputSchema", None)
        if schema is None and isinstance(t, Mapping):
            schema = t.get("inputSchema") or t.get("input_schema")
        if not isinstance(schema, Mapping):
            schema = {"type": "object", "properties": {}}

        defs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema,
                },
            }
        )
    return defs


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


def _normalize_tool_args(args: object) -> tuple[dict, bool]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {}, False

    if isinstance(args, Mapping):
        return dict(args), True

    return {}, args is None


def _tool_call_name(call: object) -> str:
    function = getattr(call, "function", None)
    name = getattr(function, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(function, Mapping):
        mapped_name = function.get("name")
        if isinstance(mapped_name, str):
            return mapped_name
    if isinstance(call, Mapping):
        mapped_fn = call.get("function")
        if isinstance(mapped_fn, Mapping):
            mapped_name = mapped_fn.get("name")
            if isinstance(mapped_name, str):
                return mapped_name
    return ""


def _tool_call_arguments(call: object) -> object:
    function = getattr(call, "function", None)
    args = getattr(function, "arguments", None)
    if args is not None:
        return args
    if isinstance(function, Mapping):
        return function.get("arguments")
    if isinstance(call, Mapping):
        mapped_fn = call.get("function")
        if isinstance(mapped_fn, Mapping):
            return mapped_fn.get("arguments")
    return None


def _tool_call_id(call: object, index: int) -> str:
    call_id = getattr(call, "id", None)
    if isinstance(call_id, str) and call_id:
        return call_id
    if isinstance(call, Mapping):
        mapped_id = call.get("id")
        if isinstance(mapped_id, str) and mapped_id:
            return mapped_id
    return f"tool_call_{index}"


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
            tools = _tool_defs(raw_tools if isinstance(raw_tools, list) else [])
        except Exception:  # pragma: no cover - transport/tool-list failure path
            logger.exception(
                "Failed to list server tools. Agent will continue without server tools."
            )
            console.print(
                "[bold yellow]Warning:[/bold yellow] Failed to list MCP tools; continuing without tool access."
            )
            tools = []

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

            llm_iteration = 0
            while True:
                if llm_iteration >= MAX_TOOL_ITERATIONS_PER_TURN:
                    warning_msg = f"Stopped after {MAX_TOOL_ITERATIONS_PER_TURN} consecutive tool-call iterations."
                    logger.warning(warning_msg)
                    console.print(f"[bold yellow]Warning:[/bold yellow] {warning_msg}")
                    assistant_message = {"role": "assistant", "content": warning_msg}
                    history.append(assistant_message)
                    _persist_message(session_id, turn, assistant_message)
                    turn += 1
                    break
                llm_iteration += 1

                llm_start = time.perf_counter()
                logger.info(
                    "LLM request started model=%s history_len=%d tools_available=%d iteration=%d",
                    CHAT_MODEL,
                    len(history),
                    len(tools),
                    llm_iteration,
                )
                try:
                    response = ollama_client.chat(model=CHAT_MODEL, messages=history, tools=tools)
                except Exception as exc:  # pragma: no cover - network/service failure path
                    console.print(f"[bold red]Ollama request failed:[/bold red] {exc}")
                    logger.error("LLM request failed model=%s error=%s", CHAT_MODEL, exc)
                    break

                msg = response.message
                llm_duration_ms = int((time.perf_counter() - llm_start) * 1000)
                logger.info(
                    "LLM request completed model=%s duration_ms=%d content_length=%d tool_call_count=%d",
                    CHAT_MODEL,
                    llm_duration_ms,
                    len(msg.content or ""),
                    len(msg.tool_calls or []),
                )

                if not msg.tool_calls:
                    console.print("[bold green]Agent:[/bold green]")
                    console.print(Markdown(msg.content or ""))
                    assistant_message = {"role": "assistant", "content": msg.content or ""}
                    history.append(assistant_message)
                    _persist_message(session_id, turn, assistant_message)
                    turn += 1
                    break

                serializable_tool_calls: list[dict] = []
                executable_calls: list[tuple[str, dict, str, bool]] = []
                for idx, tc in enumerate(msg.tool_calls, start=1):
                    name = _tool_call_name(tc)
                    if not name:
                        logger.warning(
                            "Skipping malformed tool call with missing function name index=%d", idx
                        )
                        continue

                    raw_args = _tool_call_arguments(tc)
                    args, args_valid = _normalize_tool_args(raw_args)
                    tool_call_id = _tool_call_id(tc, idx)
                    serializable_tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                    )
                    executable_calls.append((name, args, tool_call_id, args_valid))

                if not executable_calls:
                    fallback_content = (
                        msg.content or "No executable tool calls were provided by the model."
                    )
                    logger.warning(
                        "Assistant returned non-executable tool calls; continuing with fallback response."
                    )
                    console.print("[bold green]Agent:[/bold green]")
                    console.print(Markdown(fallback_content))
                    assistant_message = {"role": "assistant", "content": fallback_content}
                    history.append(assistant_message)
                    _persist_message(session_id, turn, assistant_message)
                    turn += 1
                    break

                tool_call_message = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": serializable_tool_calls,
                }
                history.append(tool_call_message)
                _persist_message(
                    session_id,
                    turn,
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                    },
                    extra_metadata={
                        "event": "tool_calls",
                        "tool_calls_json": json.dumps(
                            [
                                {
                                    "id": call["id"],
                                    "name": call["function"]["name"],
                                    "arguments": call["function"]["arguments"],
                                }
                                for call in serializable_tool_calls
                            ]
                        ),
                    },
                )
                turn += 1

                for name, args, tool_call_id, args_valid in executable_calls:
                    if not args_valid:
                        warning_msg = f"Tool call arguments for '{name}' were malformed; using empty arguments."
                        console.print(f"[bold yellow]Warning:[/bold yellow] {warning_msg}")
                        logger.warning(
                            "Tool call arguments were not valid JSON object; using empty arguments "
                            "name=%s tool_call_id=%s",
                            name,
                            tool_call_id,
                        )

                    console.print(f"[dim] > {name}({args})[/dim]")
                    logger.info(
                        "Tool call started name=%s tool_call_id=%s args_keys=%s",
                        name,
                        tool_call_id,
                        sorted(args.keys()),
                    )

                    start = time.perf_counter()
                    try:
                        result = await session.call_tool(name, args)
                        duration_ms = int((time.perf_counter() - start) * 1000)
                        result_text = _tool_result_text(result)
                        logger.info(
                            "Tool call succeeded name=%s tool_call_id=%s duration_ms=%d result_length=%d",
                            name,
                            tool_call_id,
                            duration_ms,
                            len(result_text),
                        )
                    except Exception as exc:  # pragma: no cover - transport/tool failure path
                        duration_ms = int((time.perf_counter() - start) * 1000)
                        logger.error(
                            "Tool call failed name=%s tool_call_id=%s duration_ms=%d error=%s",
                            name,
                            tool_call_id,
                            duration_ms,
                            exc,
                        )
                        result_text = f"Tool call failed: {exc}"

                    tool_message = {"role": "tool", "content": result_text, "name": name}
                    history.append(tool_message)
                    _persist_message(session_id, turn, tool_message)
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
