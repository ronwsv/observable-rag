"""Retriever: embed a query and fetch the most similar chunks from the vector store."""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk
from .providers import EmbeddingProvider
from .store import VectorStore


@dataclass
class RetrievedChunk:
    """A retrieved chunk together with its similarity score."""

    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingProvider) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed([query])[0]
        results = self.store.search(query_vector, k=k)
        return [RetrievedChunk(chunk=chunk, score=score) for chunk, score in results]
