"""Shared fixtures. Everything runs offline with hash embeddings + a mock LLM."""

from __future__ import annotations

import pytest

from gaprag.config import get_settings
from gaprag.ingest import load_corpus
from gaprag.providers import HashEmbedding, MockLLM
from gaprag.rag import RAGPipeline
from gaprag.retriever import Retriever
from gaprag.store import VectorStore


@pytest.fixture(scope="session")
def embedder() -> HashEmbedding:
    return HashEmbedding()


@pytest.fixture(scope="session")
def store(embedder: HashEmbedding) -> VectorStore:
    settings = get_settings()
    chunks = load_corpus(
        settings.corpus_dir, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    vectors = embedder.embed([c.embedding_text for c in chunks])
    return VectorStore(vectors, chunks, embedding_model=embedder.model)


@pytest.fixture()
def pipeline(store: VectorStore, embedder: HashEmbedding) -> RAGPipeline:
    return RAGPipeline(Retriever(store, embedder), MockLLM(), top_k=4)
