"""Vector store: an exact inner-product index over normalized embeddings.

Uses FAISS (``IndexFlatIP``) when available and transparently falls back to a NumPy
dot-product search otherwise, so the project runs everywhere while showcasing the
FAISS path it is designed around. The index is derived data: it is always
reproducible from the corpus via :mod:`gaprag.ingest`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chunking import Chunk


class VectorStore:
    """Holds chunk metadata and their embeddings, and searches by cosine similarity."""

    def __init__(self, vectors: np.ndarray, chunks: list[Chunk], embedding_model: str = "") -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.chunks = chunks
        self.dim = int(vectors.shape[1]) if len(vectors) else 0
        self.embedding_model = embedding_model
        self._faiss = self._build_faiss()

    # -- backend ---------------------------------------------------------------
    def _build_faiss(self):
        try:
            import faiss  # type: ignore
        except Exception:  # pragma: no cover - optional acceleration
            return None
        if not len(self.vectors):
            return None
        index = faiss.IndexFlatIP(self.dim)
        index.add(self.vectors)
        return index

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def __len__(self) -> int:
        return len(self.chunks)

    # -- search ----------------------------------------------------------------
    def search(self, query_vector: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        """Return the ``k`` most similar chunks with their inner-product scores."""
        if not len(self.chunks):
            return []
        k = min(k, len(self.chunks))
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)

        if self._faiss is not None:
            scores, idxs = self._faiss.search(query, k)
            pairs = zip(idxs[0].tolist(), scores[0].tolist(), strict=False)
        else:
            sims = (self.vectors @ query[0]).astype(float)
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top])]
            pairs = ((int(i), float(sims[i])) for i in top)

        return [(self.chunks[i], round(float(s), 4)) for i, s in pairs if i >= 0]

    # -- persistence -----------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "count": len(self.chunks),
                    "dim": self.dim,
                    "embedding_model": self.embedding_model,
                    "backend": self.backend,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        directory = Path(directory)
        vectors_path = directory / "vectors.npy"
        chunks_path = directory / "chunks.json"
        if not vectors_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No index found in {directory}. Build one first: `gaprag ingest`."
            )
        vectors = np.load(vectors_path)
        chunks = [Chunk.from_dict(d) for d in json.loads(chunks_path.read_text(encoding="utf-8"))]
        meta = {}
        meta_path = directory / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(vectors, chunks, embedding_model=meta.get("embedding_model", ""))
