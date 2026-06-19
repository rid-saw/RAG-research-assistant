"""
MCP server for the Universal RAG backend.

Exposes the backend's library + chat capabilities as MCP tools so that
Claude Desktop, Cursor, and other MCP clients can plug into the RAG.

Defaults to the local backend at http://localhost:8000. Override via
the BACKEND_URL environment variable.

Example Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "universal-rag": {
          "command": "uv",
          "args": ["--directory", "/path/to/mcp_server", "run", "python", "server.py"],
          "env": { "BACKEND_URL": "http://localhost:8000" }
        }
      }
    }
"""

import asyncio
import os

import requests
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
SHORT_TIMEOUT = 30
LONG_TIMEOUT = 120

server: Server = Server("universal-rag")


# ---------- response formatters ----------

def _format_libraries(data: dict) -> str:
    libs = data.get("libraries", [])
    if not libs:
        return "No libraries on the backend yet. Use the create_library tool to make one."
    lines = ["Available libraries:"]
    for lib in libs:
        lines.append(f"  - {lib['name']} ({lib['document_count']} chunks)")
    return "\n".join(lines)


def _format_search(data: dict) -> str:
    hits = data.get("hits", [])
    query = data.get("query", "")
    if not hits:
        return f"No hits for query: {query}"
    lines = [f"Found {len(hits)} hits for: {query}\n"]
    for i, h in enumerate(hits, start=1):
        src = h.get("source", "unknown")
        page = h.get("page")
        label = f"{src} (p.{page})" if page is not None else src
        lines.append(f"[{i}] {label}")
        lines.append(h.get("content", ""))
        lines.append("")
    return "\n".join(lines)


def _format_ask(data: dict) -> str:
    status = data.get("status", "")
    parts = [f"Status: {status}"]
    if data.get("rewritten_query"):
        parts.append(f"Rewritten query (for web): {data['rewritten_query']}")
    if data.get("reason"):
        parts.append(f"Reason: {data['reason']}")
    if data.get("answer"):
        parts.append(f"\nAnswer:\n{data['answer']}")
    if data.get("citations"):
        parts.append("\nCitations:")
        for i, c in enumerate(data["citations"], start=1):
            if c.get("page") is not None:
                label = f"{c.get('source')} (p.{c['page']})"
            elif c.get("title"):
                label = f"{c['title']} — {c.get('source')}"
            else:
                label = c.get("source", "unknown")
            parts.append(f"  {i}. {label}")
    if status == "needs_web_search":
        parts.append(
            "\nTo retry with web search, call ask_library again with the same query "
            "and allow_web_search=true. Confirm with the user first if appropriate."
        )
    return "\n".join(parts)


def _text(s: str) -> TextContent:
    return TextContent(type="text", text=s)


# ---------- tool definitions ----------

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_libraries",
            description="List every library on the RAG backend along with chunk counts.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="create_library",
            description=(
                "Create a new empty library on the backend. Library names must be "
                "alphanumeric (dashes and underscores allowed)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The library name (e.g., 'ml-papers').",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="search_library",
            description=(
                "Search a library for relevant passages. Returns the top reranked "
                "chunks without LLM synthesis — useful when you want to inspect raw "
                "retrieval before deciding how to use it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "library": {
                        "type": "string",
                        "description": "Library to search. Defaults to 'default'.",
                        "default": "default",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many passages to return (1-50).",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ask_library",
            description=(
                "Ask a question of a library and get a grounded answer with citations. "
                "Runs the corrective RAG pipeline: retrieves, grades retrieval quality, "
                "and either answers or returns status='needs_web_search'. Pass "
                "allow_web_search=true (after asking the user) to authorize the web fallback."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "library": {
                        "type": "string",
                        "description": "Library to query. Defaults to 'default'.",
                        "default": "default",
                    },
                    "allow_web_search": {
                        "type": "boolean",
                        "description": (
                            "If true, fall back to web search when the library is "
                            "insufficient. Ask the user before setting this."
                        ),
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    args = arguments or {}
    try:
        if name == "list_libraries":
            r = requests.get(f"{BACKEND_URL}/api/libraries", timeout=SHORT_TIMEOUT)
            r.raise_for_status()
            return [_text(_format_libraries(r.json()))]

        if name == "create_library":
            r = requests.post(
                f"{BACKEND_URL}/api/libraries",
                json={"name": args["name"]},
                timeout=SHORT_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            return [_text(f"Created library: {data['name']} ({data['document_count']} chunks)")]

        if name == "search_library":
            payload = {
                "query": args["query"],
                "collection": args.get("library", "default"),
            }
            if "top_k" in args:
                payload["top_k"] = args["top_k"]
            r = requests.post(
                f"{BACKEND_URL}/api/search",
                json=payload,
                timeout=LONG_TIMEOUT,
            )
            r.raise_for_status()
            return [_text(_format_search(r.json()))]

        if name == "ask_library":
            r = requests.post(
                f"{BACKEND_URL}/api/chat",
                json={
                    "query": args["query"],
                    "collection": args.get("library", "default"),
                    "allow_web_search": args.get("allow_web_search", False),
                },
                timeout=LONG_TIMEOUT,
            )
            r.raise_for_status()
            return [_text(_format_ask(r.json()))]

        return [_text(f"Unknown tool: {name}")]

    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = e.response.text or str(e)
        return [_text(f"Backend error ({e.response.status_code}): {detail}")]
    except requests.ConnectionError:
        return [_text(
            f"Could not reach backend at {BACKEND_URL}. "
            f"Is it running? Set BACKEND_URL to override."
        )]
    except Exception as e:
        return [_text(f"Error: {e}")]


# ---------- entry point ----------

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="universal-rag",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
