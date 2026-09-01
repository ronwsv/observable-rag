"""Embedding backends.

- ``sentence_transformers``: local, free, semantic (recommended for real use).
- ``openai``: hosted embeddings via the OpenAI API.
- ``hash``: dependency-free, deterministic hashed bag-of-words. It needs no model
  download, so it powers the tests and CI and gives a genuine (lexical) retrieval
  baseline — swap in a semantic backend for production-quality recall.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from .base import EmbeddingProvider, l2_normalize

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small English stop list so generic words don't dominate the lexical baseline.
_STOPWORDS = frozenset(
    """a an and are as at be by for from how if in into is it its of on or that the
    their them then there these they this to use used using was what when which who
    why will with you your do does can should would could""".split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


class HashEmbedding:
    """Deterministic hashed bag-of-words embedding (the "feature hashing" trick).

    Each token is hashed into one of ``dim`` buckets; the vector counts token hits
    and is L2-normalized. No training, no downloads, fully reproducible.
    """

    name = "hash"

    def __init__(self, dim: int = 512, model: str = "hash-bow") -> None:
        self.dim = dim
        self.model = model

    def _hash(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324 - not security
        return int.from_bytes(digest[:4], "little") % self.dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _tokenize(text):
                out[i, self._hash(token)] += 1.0
        # Sublinear term-frequency scaling (1 + log(tf)) damps repeated tokens so
        # distinctive terms carry more weight, then L2-normalize for cosine search.
        np.log1p(out, out=out)
        return l2_normalize(out)


class SentenceTransformerEmbedding:
    """Local semantic embeddings via ``sentence-transformers`` (the ``[local]`` extra)."""

    name = "sentence_transformers"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is not installed. Install the extra:\n"
                '    pip install -e ".[local]"\n'
                "or set GAPRAG_EMBEDDING_PROVIDER=hash for a dependency-free baseline."
            ) from exc
        self.model = model
        self._encoder = SentenceTransformer(model)
        self.dim = int(self._encoder.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._encoder.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)


class OpenAIEmbedding:
    """Hosted embeddings via the OpenAI API (the ``[openai]`` extra + ``OPENAI_API_KEY``)."""

    name = "openai"
    _DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError('OpenAI SDK missing. Install: pip install -e ".[openai]"') from exc
        self.model = model
        self._client = OpenAI()
        self.dim = self._DIMS.get(model, 1536)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.model, input=list(texts))
        vectors = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
        return l2_normalize(vectors)


class BedrockEmbedding:
    """Amazon Bedrock embeddings, e.g. Titan (the ``[bedrock]`` extra + AWS creds)."""

    name = "bedrock"
    _DIMS = {"amazon.titan-embed-text-v2:0": 1024, "amazon.titan-embed-text-v1": 1536}

    def __init__(
        self, model: str = "amazon.titan-embed-text-v2:0", region: str = "us-east-1"
    ) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError('boto3 missing. Install: pip install -e ".[bedrock]"') from exc
        self.model = model
        self.dim = self._DIMS.get(model, 1024)
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import json

        # Titan embeds one text per call; batch by looping.
        vectors = []
        for text in texts:
            resp = self._client.invoke_model(
                modelId=self.model, body=json.dumps({"inputText": text})
            )
            vectors.append(json.loads(resp["body"].read())["embedding"])
        return l2_normalize(np.asarray(vectors, dtype=np.float32))


def build_embedding_provider(
    provider: str, model: str, *, region: str = "us-east-1"
) -> EmbeddingProvider:
    """Factory: map a provider name from settings to an embedding backend."""
    provider = provider.lower()
    if provider == "hash":
        return HashEmbedding()
    if provider in {"sentence_transformers", "st", "local"}:
        return SentenceTransformerEmbedding(model=model)
    if provider == "openai":
        return OpenAIEmbedding(model=model)
    if provider == "bedrock":
        return BedrockEmbedding(model=model, region=region)
    raise ValueError(f"Unknown embedding provider: {provider!r}")
