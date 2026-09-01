"""Tests for the optional LangChain retrieval backend.

Skipped automatically when the [langchain] extra is not installed, so CI stays lean.
"""

import pytest

pytest.importorskip("langchain_community")

from gaprag.ingest import load_corpus  # noqa: E402
from gaprag.langchain_retriever import LangChainRetriever  # noqa: E402
from gaprag.providers import HashEmbedding, MockLLM  # noqa: E402
from gaprag.rag import RAGPipeline  # noqa: E402


@pytest.fixture(scope="module")
def lc_retriever():
    from gaprag.config import get_settings

    settings = get_settings()
    chunks = load_corpus(
        settings.corpus_dir, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    return LangChainRetriever(chunks, HashEmbedding())


def test_langchain_retriever_finds_relevant_document(lc_retriever):
    results = lc_retriever.retrieve("What is the FAISS IndexFlatIP index?", k=4)
    assert results
    assert any(r.chunk.source == "embeddings-and-vector-search.md" for r in results)


def test_langchain_scores_sorted_descending(lc_retriever):
    results = lc_retriever.retrieve("chunk overlap and recall", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_pipeline_runs_on_langchain_backend(lc_retriever):
    pipeline = RAGPipeline(lc_retriever, MockLLM(), top_k=4)
    resp = pipeline.answer("What is Retrieval-Augmented Generation?")
    assert resp.route == "retrieve"
    assert resp.citations
    assert resp.trace.embedding_model == "hash-bow"
