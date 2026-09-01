"""Lightweight observability: per-request tracing, structured logs, cost estimation.

Every RAG call produces one :class:`RequestTrace` capturing the route taken, the
retrieved chunk ids, per-stage latency, token usage and an estimated cost. Calling
:meth:`RequestTrace.emit` writes a single structured (JSON) log line and, if the
optional ``langfuse`` extra and keys are present, forwards the trace to Langfuse.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

logger = logging.getLogger("gaprag")


def configure_logging(json_logs: bool = True, level: int = logging.INFO) -> None:
    """Configure the ``gaprag`` logger once, idempotently."""
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    if not json_logs:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


# Rough prices in USD per 1M tokens (input, output). Estimation only — update freely.
_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate request cost. Unknown / local models (e.g. Ollama) are treated as free.

    Matches by longest price-table key contained in the model id, so both direct ids
    (``claude-3-5-haiku-20241022``) and Bedrock ids
    (``us.anthropic.claude-3-5-haiku-20241022-v1:0``) resolve correctly.
    """
    candidates = [k for k in _PRICE_PER_MTOK if k in model]
    if not candidates:
        return 0.0
    key = max(candidates, key=len)
    in_price, out_price = _PRICE_PER_MTOK[key]
    cost = prompt_tokens / 1_000_000 * in_price + completion_tokens / 1_000_000 * out_price
    return round(cost, 6)


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) for providers that do not report usage."""
    return max(1, len(text) // 4)


@dataclass
class RequestTrace:
    """Structured record of a single RAG request."""

    query: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    route: str = "retrieve"
    retrieved_ids: list[str] = field(default_factory=list)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    embedding_model: str = ""
    cost_usd: float = 0.0

    @property
    def total_ms(self) -> float:
        return round(self.retrieval_ms + self.generation_ms, 2)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def finalize(self) -> RequestTrace:
        """Compute the estimated cost from the recorded token usage."""
        self.cost_usd = estimate_cost_usd(self.model, self.prompt_tokens, self.completion_tokens)
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_ms"] = self.total_ms
        d["total_tokens"] = self.total_tokens
        return d

    def emit(self) -> None:
        """Write one structured log line and forward to Langfuse if configured."""
        logger.info(json.dumps({"event": "rag_request", **self.to_dict()}))
        _emit_langfuse(self)


def _emit_langfuse(trace: RequestTrace) -> None:
    """Best-effort Langfuse export. No-op if the extra or keys are missing."""
    try:
        import os

        if not os.getenv("LANGFUSE_PUBLIC_KEY"):
            return
        from langfuse import Langfuse  # type: ignore

        client = Langfuse()
        client.trace(
            name="rag_request",
            input=trace.query,
            metadata=trace.to_dict(),
        )
    except Exception:  # pragma: no cover - optional dependency / network
        return


@contextmanager
def timer():
    """Context manager yielding a callable that returns elapsed milliseconds."""
    start = time.perf_counter()
    elapsed = {"ms": 0.0}

    def read() -> float:
        return round((time.perf_counter() - start) * 1000, 2)

    try:
        yield read
    finally:
        elapsed["ms"] = read()
