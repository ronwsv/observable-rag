"""FastAPI service exposing the RAG pipeline.

Endpoints:
- ``GET  /health`` — liveness + whether an index is present (no model load).
- ``POST /chat``   — ask a question; returns answer, citations and the request trace.
- ``POST /ingest`` — (re)build the vector index from the corpus.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .ingest import ingest
from .observability import configure_logging
from .rag import RAGPipeline, build_pipeline

_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(json_logs=get_settings().json_logs)
    yield


app = FastAPI(
    title="Observable RAG",
    version=__version__,
    description="Evaluated, observable RAG service.",
    lifespan=lifespan,
)


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = build_pipeline()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _pipeline


# -- schemas ------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["What is RAG and when should I use it?"])
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationModel(BaseModel):
    chunk_id: str
    source: str
    heading: str
    score: float
    snippet: str


class TraceModel(BaseModel):
    request_id: str
    route: str
    retrieved_ids: list[str]
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    embedding_model: str
    cost_usd: float


class ChatResponse(BaseModel):
    answer: str
    route: str
    citations: list[CitationModel]
    trace: TraceModel


# -- endpoints ----------------------------------------------------------------
@app.get("/")
def root() -> dict:
    return {"name": "Observable RAG", "version": __version__, "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    index_ready = (settings.index_dir / "chunks.json").exists()
    return {
        "status": "ok",
        "index_ready": index_ready,
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    pipeline = get_pipeline()
    result = pipeline.answer(request.query, top_k=request.top_k)
    trace = result.trace
    return ChatResponse(
        answer=result.answer,
        route=result.route,
        citations=[CitationModel(**c.__dict__) for c in result.citations],
        trace=TraceModel(
            request_id=trace.request_id,
            route=trace.route,
            retrieved_ids=trace.retrieved_ids,
            retrieval_ms=trace.retrieval_ms,
            generation_ms=trace.generation_ms,
            total_ms=trace.total_ms,
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            total_tokens=trace.total_tokens,
            model=trace.model,
            embedding_model=trace.embedding_model,
            cost_usd=trace.cost_usd,
        ),
    )


@app.post("/ingest")
def rebuild_index() -> dict:
    global _pipeline
    stats = ingest()
    _pipeline = None  # force rebuild on next request so the new index is picked up
    return stats.__dict__
