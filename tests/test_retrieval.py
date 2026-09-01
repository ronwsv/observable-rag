import numpy as np

from gaprag.providers import HashEmbedding
from gaprag.retriever import Retriever


def test_embeddings_are_unit_normalized():
    emb = HashEmbedding()
    vecs = emb.embed(["hello world", "faiss vector search"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_retriever_finds_relevant_document(store, embedder):
    retriever = Retriever(store, embedder)
    results = retriever.retrieve("What is the FAISS IndexFlatIP index?", k=4)
    assert results
    assert any(r.chunk.source == "embeddings-and-vector-search.md" for r in results)


def test_scores_are_sorted_descending(store, embedder):
    retriever = Retriever(store, embedder)
    results = retriever.retrieve("chunk overlap", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
