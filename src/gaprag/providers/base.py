"""Provider protocols and shared result types.

Both LLM and embedding backends are duck-typed against these protocols, which keeps
the rest of the codebase provider-agnostic: swapping Ollama for OpenAI is a config
change, not a code change.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class LLMResult:
    """The text a model returned plus its token usage (for cost/latency tracing)."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into unit-normalized dense vectors (so inner product == cosine)."""

    name: str
    model: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape ``(len(texts), dim)``, L2-normalized."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Generates a completion from a system + user prompt."""

    name: str
    model: str

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult:
        ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows, guarding against division by zero."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
