"""Model Context Protocol (MCP) server exposing the RAG service as tools.

MCP is a standard way for an LLM host (Claude Desktop, IDEs, agents) to call
external tools in a controlled way. This server turns Observable RAG into two MCP
tools any compatible client can use:

- ``search_knowledge_base(query, k)`` — semantic retrieval over the corpus.
- ``answer_question(query)`` — the full RAG pipeline (route → retrieve → generate),
  returning the answer with citations.

The pipeline is built lazily on first use, so importing this module (and running the
test suite) needs no index or API keys. Run it with ``gaprag mcp`` and point an MCP
client at that command. Needs the ``[mcp]`` extra and a built index (``gaprag ingest``).
"""

from __future__ import annotations

from typing import Any

# FastMCP was renamed to MCPServer in mcp 2.x; support both so the server runs on
# whichever version is installed.
try:
    from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('MCP SDK missing. Install: pip install -e ".[mcp]"') from exc

from .rag import RAGPipeline, build_pipeline

server = _Server("observable-rag")

_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


@server.tool()
def search_knowledge_base(query: str, k: int = 4) -> list[dict[str, Any]]:
    """Search the knowledge base and return the most relevant passages with scores."""
    results = _get_pipeline().retriever.retrieve(query, k=k)
    return [
        {
            "id": r.chunk.id,
            "source": r.chunk.source,
            "heading": r.chunk.heading,
            "score": r.score,
            "text": r.chunk.text,
        }
        for r in results
    ]


@server.tool()
def answer_question(query: str) -> dict[str, Any]:
    """Answer a question using the full RAG pipeline, returning the answer and citations."""
    response = _get_pipeline().answer(query)
    return {
        "answer": response.answer,
        "route": response.route,
        "citations": [
            {"source": c.source, "heading": c.heading, "score": c.score} for c in response.citations
        ],
    }


def main() -> None:
    """Run the MCP server over stdio (the transport MCP clients expect)."""
    server.run()


if __name__ == "__main__":
    main()
