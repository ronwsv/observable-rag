"""Tests for the LangGraph agent orchestrator. Skipped when langgraph is absent."""

import pytest

pytest.importorskip("langgraph")

from gaprag.graph import build_rag_graph  # noqa: E402
from gaprag.providers import MockLLM  # noqa: E402
from gaprag.retriever import Retriever  # noqa: E402


def test_langgraph_agent_answers(store, embedder):
    agent = build_rag_graph(Retriever(store, embedder), MockLLM())
    final = agent.invoke({"query": "What is RAG and when should I use it?"})
    assert final["route"] == "retrieve"
    assert final.get("answer")
    assert final.get("citations")


def test_langgraph_direct_route_skips_retrieval(store, embedder):
    agent = build_rag_graph(Retriever(store, embedder), MockLLM())
    final = agent.invoke({"query": "hello there"})
    assert final["route"] == "direct"
    assert final.get("answer")
    assert not final.get("citations")
