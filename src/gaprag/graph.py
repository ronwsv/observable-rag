"""LangGraph orchestration of the agentic RAG flow.

The same route -> retrieve -> generate logic as :mod:`gaprag.rag`, expressed as an
explicit LangGraph state machine: a ``route`` node decides the path, a conditional
edge sends the query either through ``retrieve`` (then ``generate``) or straight to
``generate``. This is the graph-of-nodes model LangGraph formalizes for more complex,
multi-step and multi-agent flows — here kept small and readable on purpose.

Reuses the existing retriever, router and LLM unchanged, so it is a drop-in
orchestrator, not a rewrite. Needs the ``[langgraph]`` extra.
"""

from __future__ import annotations

from typing import TypedDict

from .providers import LLMProvider
from .rag import ANSWER_SYSTEM, DIRECT_SYSTEM
from .retriever import Retriever
from .router import Router


class RagState(TypedDict, total=False):
    """State threaded through the graph; each node returns a partial update."""

    query: str
    route: str
    context: str
    citations: list[dict]
    answer: str


def build_rag_graph(
    retriever: Retriever, llm: LLMProvider, router: Router | None = None, top_k: int = 4
):
    """Build and compile the LangGraph agent for the RAG flow."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('LangGraph missing. Install: pip install -e ".[langgraph]"') from exc

    router = router or Router(llm)

    def route_node(state: RagState) -> dict:
        return {"route": router.route(state["query"]).action}

    def retrieve_node(state: RagState) -> dict:
        retrieved = retriever.retrieve(state["query"], k=top_k)
        context = "\n\n".join(
            f"[{rc.chunk.id}] ({rc.chunk.heading})\n{rc.chunk.text}" for rc in retrieved
        )
        citations = [
            {"chunk_id": rc.chunk.id, "source": rc.chunk.source, "score": rc.score}
            for rc in retrieved
        ]
        return {"context": context, "citations": citations}

    def generate_node(state: RagState) -> dict:
        if state.get("route") == "retrieve":
            user = f"Context:\n{state.get('context', '')}\n\nQuestion: {state['query']}"
            result = llm.complete(ANSWER_SYSTEM, user)
        else:
            result = llm.complete(DIRECT_SYSTEM, state["query"])
        return {"answer": result.text}

    def route_condition(state: RagState) -> str:
        return "retrieve" if state.get("route") == "retrieve" else "generate"

    builder = StateGraph(RagState)
    builder.add_node("route", route_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route", route_condition, {"retrieve": "retrieve", "generate": "generate"}
    )
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()
