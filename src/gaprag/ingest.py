"""Ingestion pipeline: corpus -> chunks -> embeddings -> saved vector index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, chunk_markdown
from .config import Settings, get_settings
from .providers import EmbeddingProvider, build_embedding_provider
from .store import VectorStore


@dataclass
class IngestStats:
    documents: int
    chunks: int
    dim: int
    backend: str
    embedding_model: str


def load_corpus(corpus_dir: str | Path, *, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Read every ``.md``/``.txt`` file under ``corpus_dir`` and chunk it."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    chunks: list[Chunk] = []
    files = sorted(p for p in corpus_dir.rglob("*") if p.suffix.lower() in {".md", ".txt"})
    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks.extend(
            chunk_markdown(text, path.name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    return chunks


def ingest(
    settings: Settings | None = None, embedder: EmbeddingProvider | None = None
) -> IngestStats:
    """Build and persist the vector index from the configured corpus."""
    settings = settings or get_settings()
    embedder = embedder or build_embedding_provider(
        settings.embedding_provider, settings.embedding_model, region=settings.aws_region
    )

    chunks = load_corpus(
        settings.corpus_dir, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    if not chunks:
        raise ValueError(f"No documents found in {settings.corpus_dir}")

    vectors = embedder.embed([c.embedding_text for c in chunks])
    store = VectorStore(vectors, chunks, embedding_model=embedder.model)
    store.save(settings.index_dir)

    documents = len({c.source for c in chunks})
    return IngestStats(
        documents=documents,
        chunks=len(chunks),
        dim=store.dim,
        backend=store.backend,
        embedding_model=embedder.model,
    )
