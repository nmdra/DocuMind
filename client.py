from __future__ import annotations

import asyncio
import json

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console
from rich.markdown import Markdown

from config import CHAT_MODEL, EMBED_MODEL, OLLAMA_BASE_URL

console = Console()
ollama_client = ollama.Client(host=OLLAMA_BASE_URL)


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


async def run_agent() -> None:
    params = StdioServerParameters(
        command="python3", args=["-m", "uv", "run", "python", "server.py"]
    )

    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tool_resp = await session.list_tools()
        tools = _tool_defs(tool_resp.tools)

        console.print("[bold green]FastMCP + ChromaDB agent ready.[/bold green]")
        console.print(f"[dim]{CHAT_MODEL} | {EMBED_MODEL}[/dim]")
        console.print("[dim]Type 'quit' to exit.[/dim]")

        history: list[dict] = []
        while True:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                break

            history.append({"role": "user", "content": user_input})

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
                    history.append({"role": "assistant", "content": msg.content or ""})
                    break

                history.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": msg.tool_calls,
                    }
                )

                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = _normalize_tool_args(tc.function.arguments)

                    console.print(f"[dim] > {name}({args})[/dim]")

                    try:
                        result = await session.call_tool(name, args)
                        result_text = result.content[0].text if result.content else ""
                    except Exception as exc:  # pragma: no cover - transport/tool failure path
                        result_text = f"Tool call failed: {exc}"

                    history.append({"role": "tool", "content": result_text, "name": name})


if __name__ == "__main__":
    asyncio.run(run_agent())
