"""Optional LangChain retrieval backend.

This is the same retrieval step as :mod:`gaprag.retriever`, but built on LangChain's
FAISS vectorstore instead of calling FAISS directly. It exists to demonstrate
familiarity with the LangChain ecosystem while proving a design point: because the
pipeline depends only on a small ``retrieve(query, k) -> [RetrievedChunk]`` contract,
the framework is a swappable detail, not the architecture.

The project's own :class:`~gaprag.providers` embeddings are reused unchanged — wrapped
in a thin LangChain ``Embeddings`` adapter — so both backends share the exact same
vectors and are directly comparable.

Enable with ``GAPRAG_RETRIEVER=langchain`` (needs the ``[langchain]`` extra).
"""

from __future__ import annotations

from .chunking import Chunk
from .providers import EmbeddingProvider
from .retriever import RetrievedChunk


def _wrap_embeddings(provider: EmbeddingProvider):
    """Adapt an Observable RAG EmbeddingProvider to LangChain's ``Embeddings`` interface."""
    from langchain_core.embeddings import Embeddings

    class _ProviderEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return provider.embed(texts).tolist()

        def embed_query(self, text: str) -> list[float]:
            return provider.embed([text])[0].tolist()

    return _ProviderEmbeddings()


class LangChainRetriever:
    """Retriever backed by LangChain's FAISS vectorstore, sharing our embeddings.

    Exposes the same ``embedder`` attribute and ``retrieve`` method as
    :class:`~gaprag.retriever.Retriever`, so :class:`~gaprag.rag.RAGPipeline` uses it
    without any changes.
    """

    def __init__(self, chunks: list[Chunk], embedder: EmbeddingProvider) -> None:
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "LangChain backend requested but not installed. Install the extra:\n"
                '    pip install -e ".[langchain]"\n'
                "or set GAPRAG_RETRIEVER=native to use the built-in FAISS store."
            ) from exc

        self.embedder = embedder
        metadatas = [
            {"id": c.id, "source": c.source, "heading": c.heading, "text": c.text} for c in chunks
        ]
        self._store = FAISS.from_texts(
            texts=[c.embedding_text for c in chunks],
            embedding=_wrap_embeddings(embedder),
            metadatas=metadatas,
        )

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed([query])[0].tolist()
        docs_and_scores = self._store.similarity_search_with_score_by_vector(query_vector, k=k)
        results = []
        for doc, distance in docs_and_scores:
            meta = doc.metadata
            chunk = Chunk(
                id=meta["id"], source=meta["source"], heading=meta["heading"], text=meta["text"]
            )
            # FAISS returns the *squared* L2 distance. For L2-normalized vectors,
            # ||a-b||^2 = 2 - 2*cos, so cosine = 1 - distance/2. This maps LangChain's
            # score back to the same scale the native inner-product store reports.
            score = round(1.0 - float(distance) / 2.0, 4)
            results.append(RetrievedChunk(chunk=chunk, score=score))
        return results
