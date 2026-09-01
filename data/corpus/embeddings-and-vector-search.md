# Embeddings and Vector Search

An **embedding** is a dense vector that represents the meaning of a piece of text.
Texts with similar meaning map to vectors that are close together in the vector
space, which is what makes semantic search possible: we can find passages that are
*relevant* to a query even when they share no exact keywords.

## Similarity metrics

- **Cosine similarity** measures the angle between two vectors and ignores their
  magnitude. It is the default choice for text embeddings.
- **Dot product** (inner product) equals cosine similarity when the vectors are
  L2-normalized. Normalizing embeddings and using inner product is a common,
  efficient setup.
- **Euclidean (L2) distance** is also available but less common for text.

Because cosine similarity equals the dot product of normalized vectors, a typical
pipeline normalizes every embedding to unit length and then uses an inner-product
index.

## FAISS

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search
over dense vectors. The simplest index, `IndexFlatIP`, does an exact brute-force
inner-product search — perfect for small to medium corpora because it is exact and
has no training step. For millions of vectors, approximate indexes such as `IVF`
(inverted file) or `HNSW` (graph-based) trade a little recall for much faster
search and lower memory.

A minimal FAISS setup for RAG:

1. Embed and L2-normalize every chunk.
2. Add the vectors to an `IndexFlatIP` of the right dimension.
3. At query time, embed and normalize the query, then call `index.search(q, k)`
   to get the top-k nearest chunks and their inner-product scores.

## Choosing an embedding model

Small models such as `all-MiniLM-L6-v2` (384 dimensions) or `bge-small-en` are fast
and cheap and are strong baselines. Larger models improve recall on hard queries at
higher cost and latency. Always evaluate embedding models on *your own* data and
queries — leaderboard rank does not guarantee the best result for your corpus.

## Sparse vs dense retrieval

Dense embeddings capture semantics; sparse lexical methods such as BM25 capture
exact term matches. **Hybrid search** combines both and often beats either alone,
especially for queries with rare keywords, product codes, or names.
