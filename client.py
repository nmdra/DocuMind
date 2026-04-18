from __future__ import annotations

import argparse
import asyncio
import json
import shlex
from collections.abc import Mapping

import chromadb
import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from rich.console import Console
from rich.markdown import Markdown

from config import (
    CHAT_MODEL,
    CHROMA_PATH,
    CONVERSATION_COLLECTION_NAME,
    DEFAULT_CONVERSATION_SESSION_ID,
    DEFAULT_SERVER_COMMAND,
    DEFAULT_TRANSPORT,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    SSE_HOST,
    SSE_PORT,
)

console = Console()
ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
chroma = chromadb.PersistentClient(path=CHROMA_PATH)
conversation_col = chroma.get_or_create_collection(
    name=CONVERSATION_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
EMPTY_MESSAGE_SENTINEL = "__EMPTY_MESSAGE__"


def _tool_defs(tools) -> list[dict]:
    """Convert MCP tools into Ollama function-tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,
            },
        }
        for t in tools
    ]


def _normalize_tool_args(args: object) -> dict:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {}

    if isinstance(args, dict):
        return args

    return {}


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

        params = StdioServerParameters(command=parts[0], args=parts[1:])
        transport_context = stdio_client(params)
    elif transport == "sse":
        transport_context = sse_client(sse_url)
    else:
        raise ValueError(f"Unsupported transport: {transport!r}. Expected one of: 'stdio', 'sse'.")

    async with (
        transport_context as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tool_resp = await session.list_tools()
        tools = _tool_defs(tool_resp.tools)

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

            while True:
                try:
                    response = ollama_client.chat(model=CHAT_MODEL, messages=history, tools=tools)
                except Exception as exc:  # pragma: no cover - network/service failure path
                    console.print(f"[bold red]Ollama request failed:[/bold red] {exc}")
                    break

                msg = response.message

                if not msg.tool_calls:
                    console.print("[bold green]Agent:[/bold green]")
                    console.print(Markdown(msg.content or ""))
                    assistant_message = {"role": "assistant", "content": msg.content or ""}
                    history.append(assistant_message)
                    _persist_message(session_id, turn, assistant_message)
                    turn += 1
                    break

                tool_call_message = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
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
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                }
                                for call in msg.tool_calls
                            ]
                        ),
                    },
                )
                turn += 1

                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = _normalize_tool_args(tc.function.arguments)

                    console.print(f"[dim] > {name}({args})[/dim]")

                    try:
                        result = await session.call_tool(name, args)
                        result_text = result.content[0].text if result.content else ""
                    except Exception as exc:  # pragma: no cover - transport/tool failure path
                        result_text = f"Tool call failed: {exc}"

                    tool_message = {"role": "tool", "content": result_text, "name": name}
                    history.append(tool_message)
                    _persist_message(session_id, turn, tool_message)
                    turn += 1


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_agent(
            args.session_id,
            args.server_command,
            args.transport,
            args.sse_url,
        )
    )
