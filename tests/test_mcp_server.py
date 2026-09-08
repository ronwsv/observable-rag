"""MCP server wiring (no pipeline build). Skipped when the mcp SDK is absent."""

import pytest

pytest.importorskip("mcp")

from gaprag import mcp_server  # noqa: E402


def test_server_is_created():
    assert mcp_server.server is not None
    assert mcp_server.server.name == "observable-rag"


def test_tools_are_registered_callables():
    # FastMCP's @tool() decorator returns the function, so it stays callable here.
    assert callable(mcp_server.search_knowledge_base)
    assert callable(mcp_server.answer_question)
