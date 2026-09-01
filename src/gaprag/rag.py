"""RAG orchestration: route -> retrieve -> generate, with full request tracing.

This module ties the retriever, router and LLM together and records a
:class:`~gaprag.observability.RequestTrace` for every call (route, retrieved ids,
per-stage latency, token usage, estimated cost).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings, get_settings
from .observability import RequestTrace, timer
from .providers import EmbeddingProvider, LLMProvider, build_embedding_provider, build_llm_provider
from .retriever import Retriever
from .router import Router
from .store import VectorStore

ANSWER_SYSTEM = """You are a precise assistant that answers questions using ONLY the \
context provided. Follow these rules strictly:
- Use only the information in the context. Do not use outside knowledge.
- If the answer is not in the context, say "I don't know based on the provided context."
- Cite the source id (e.g. [rag-fundamentals.md#2]) next to each claim.
- Be concise and factual."""

DIRECT_SYSTEM = """You are a helpful assistant. Answer briefly and clearly."""


@dataclass
class Citation:
    chunk_id: str
    source: str
    heading: str
    score: float
    snippet: str


@dataclass
class RAGResponse:
    answer: str
    route: str
    citations: list[Citation] = field(default_factory=list)
    trace: RequestTrace | None = None


def _build_context(retrieved) -> str:
    blocks = []
    for rc in retrieved:
        blocks.append(f"[{rc.chunk.id}] ({rc.chunk.heading})\n{rc.chunk.text}")
    return "\n\n".join(blocks)


class RAGPipeline:
    """End-to-end RAG: an agentic router in front of grounded retrieval + generation."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        router: Router | None = None,
        top_k: int = 4,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.router = router or Router(llm)
        self.top_k = top_k

    def answer(self, query: str, top_k: int | None = None) -> RAGResponse:
        k = top_k or self.top_k
        trace = RequestTrace(
            query=query,
            model=getattr(self.llm, "model", ""),
            embedding_model=getattr(self.retriever.embedder, "model", ""),
        )

        route = self.router.route(query)
        trace.route = route.action

        citations: list[Citation] = []
        if route.action == "retrieve":
            with timer() as elapsed:
                retrieved = self.retriever.retrieve(query, k=k)
            trace.retrieval_ms = elapsed()
            trace.retrieved_ids = [rc.chunk.id for rc in retrieved]
            citations = [
                Citation(
                    chunk_id=rc.chunk.id,
                    source=rc.chunk.source,
                    heading=rc.chunk.heading,
                    score=rc.score,
                    snippet=rc.chunk.text[:160].strip(),
                )
                for rc in retrieved
            ]
            system = ANSWER_SYSTEM
            user = f"Context:\n{_build_context(retrieved)}\n\nQuestion: {query}"
        else:
            system = DIRECT_SYSTEM
            user = query

        with timer() as elapsed:
            result = self.llm.complete(system, user, temperature=0.0)
        trace.generation_ms = elapsed()
        trace.prompt_tokens = result.prompt_tokens
        trace.completion_tokens = result.completion_tokens
        trace.model = result.model or trace.model
        trace.finalize().emit()

        return RAGResponse(answer=result.text, route=route.action, citations=citations, trace=trace)


def build_pipeline(
    settings: Settings | None = None,
    *,
    embedder: EmbeddingProvider | None = None,
    llm: LLMProvider | None = None,
) -> RAGPipeline:
    """Construct a pipeline from settings, loading the persisted index from disk."""
    settings = settings or get_settings()
    embedder = embedder or build_embedding_provider(
        settings.embedding_provider, settings.embedding_model, region=settings.aws_region
    )
    llm = llm or build_llm_provider(
        settings.llm_provider,
        settings.llm_model,
        ollama_base_url=settings.ollama_base_url,
        timeout=settings.request_timeout,
        region=settings.aws_region,
    )
    store = VectorStore.load(settings.index_dir)
    if settings.retriever == "langchain":
        from .langchain_retriever import LangChainRetriever

        retriever: Retriever | LangChainRetriever = LangChainRetriever(store.chunks, embedder)
    else:
        retriever = Retriever(store, embedder)
    return RAGPipeline(retriever, llm, Router(llm), top_k=settings.top_k)
