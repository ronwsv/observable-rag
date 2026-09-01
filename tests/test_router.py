from gaprag.providers import MockLLM
from gaprag.router import Route, Router, _heuristic


def test_router_routes_factual_query_to_retrieve():
    route = Router(MockLLM()).route("What is agentic RAG?")
    assert route.action == "retrieve"


def test_router_routes_greeting_to_direct():
    route = Router(MockLLM()).route("hello there")
    assert route.action == "direct"


def test_heuristic_fallback_defaults_to_retrieve():
    assert _heuristic("explain embeddings").action == "retrieve"
    assert _heuristic("thanks!").action == "direct"


def test_router_survives_bad_llm_output():
    class BrokenLLM:
        model = "broken"

        def complete(self, system, user, temperature=0.0):
            raise RuntimeError("boom")

    route = Router(BrokenLLM()).route("What is RAG?")
    assert isinstance(route, Route)
    assert route.action == "retrieve"
