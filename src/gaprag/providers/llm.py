"""LLM backends: Ollama (local default), OpenAI, Anthropic, and a deterministic Mock.

All backends share the same :class:`LLMResult` return type and report token usage
when the provider exposes it, so cost and latency tracing works uniformly.
"""

from __future__ import annotations

import json
import re

import httpx

from ..observability import approx_tokens
from .base import LLMResult

_SMALLTALK = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|bye|good (morning|evening))\b", re.I)


class MockLLM:
    """Offline, deterministic backend for tests, CI and no-key demos.

    It never calls the network. For routing prompts it returns valid JSON; for
    answering prompts it extracts a grounded snippet from the provided context.
    """

    name = "mock"

    def __init__(self, model: str = "mock") -> None:
        self.model = model

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult:
        if '"action"' in system:  # routing prompt
            action = "direct" if _SMALLTALK.match(user) else "retrieve"
            text = json.dumps({"action": action, "reason": "mock heuristic decision"})
        else:  # answering prompt
            text = self._grounded_answer(user)
        return LLMResult(
            text=text,
            prompt_tokens=approx_tokens(system + user),
            completion_tokens=approx_tokens(text),
            model=self.model,
        )

    @staticmethod
    def _grounded_answer(user: str) -> str:
        """Return the first couple of sentences of the retrieved context, or a refusal."""
        marker = "Context:"
        if marker not in user:
            return "I don't have context to answer that."
        context = user.split(marker, 1)[1]
        # Drop source labels like "[rag-fundamentals.md#2]" then take two sentences.
        cleaned = re.sub(r"\[[^\]]+\]", "", context).strip()
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        snippet = " ".join(s.strip() for s in sentences[:2] if s.strip())
        return snippet or "I don't know based on the provided context."


class OllamaLLM:
    """Local models served by Ollama via its HTTP API (no API key needed)."""

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult:
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResult(
            text=data.get("message", {}).get("content", "").strip(),
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            model=self.model,
        )


class OpenAILLM:
    """OpenAI chat models (the ``[openai]`` extra + ``OPENAI_API_KEY``)."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError('OpenAI SDK missing. Install: pip install -e ".[openai]"') from exc
        self.model = model
        self._client = OpenAI()

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )
        usage = resp.usage
        return LLMResult(
            text=(resp.choices[0].message.content or "").strip(),
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            model=self.model,
        )


class AnthropicLLM:
    """Anthropic Claude models (the ``[anthropic]`` extra + ``ANTHROPIC_API_KEY``)."""

    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-haiku-20241022") -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                'Anthropic SDK missing. Install: pip install -e ".[anthropic]"'
            ) from exc
        self.model = model
        self._client = Anthropic()

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult:
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=1024,
            temperature=temperature,
        )
        text = "".join(block.text for block in resp.content if block.type == "text").strip()
        return LLMResult(
            text=text,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            model=self.model,
        )


class BedrockLLM:
    """Amazon Bedrock via the unified Converse API (the ``[bedrock]`` extra + AWS creds).

    The Converse API is model-agnostic across Bedrock (Claude, Llama, Titan, ...) and
    reports token usage, so cost/latency tracing works the same as every other backend.
    Credentials are resolved by boto3's standard chain — this code never handles keys.
    """

    name = "bedrock"

    def __init__(
        self,
        model: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        region: str = "us-east-1",
    ) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError('boto3 missing. Install: pip install -e ".[bedrock]"') from exc
        self.model = model
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult:
        resp = self._client.converse(
            modelId=self.model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"temperature": temperature, "maxTokens": 1024},
        )
        text = resp["output"]["message"]["content"][0]["text"].strip()
        usage = resp.get("usage", {})
        return LLMResult(
            text=text,
            prompt_tokens=usage.get("inputTokens", 0),
            completion_tokens=usage.get("outputTokens", 0),
            model=self.model,
        )


class GeminiLLM:
    """Google Gemini via the unified google-genai SDK (the ``[gemini]`` extra).

    Reads GOOGLE_API_KEY / GEMINI_API_KEY from the environment. The client is created
    lazily, so the provider can be constructed without a key (e.g. in tests).
    """

    name = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        try:
            from google import genai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError('google-genai missing. Install: pip install -e ".[gemini]"') from exc
        self.model = model
        self._client = None

    def _client_or_create(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client()
        return self._client

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult:
        from google.genai import types

        resp = self._client_or_create().models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=temperature
            ),
        )
        usage = resp.usage_metadata
        return LLMResult(
            text=(resp.text or "").strip(),
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            model=self.model,
        )


def build_llm_provider(
    provider: str,
    model: str,
    *,
    ollama_base_url: str = "http://localhost:11434",
    timeout: float = 60.0,
    region: str = "us-east-1",
):
    """Factory: map a provider name from settings to an LLM backend."""
    provider = provider.lower()
    if provider == "mock":
        return MockLLM(model="mock")
    if provider == "ollama":
        return OllamaLLM(model=model, base_url=ollama_base_url, timeout=timeout)
    if provider == "openai":
        return OpenAILLM(model=model)
    if provider == "anthropic":
        return AnthropicLLM(model=model)
    if provider == "bedrock":
        return BedrockLLM(model=model, region=region)
    if provider == "gemini":
        return GeminiLLM(model=model)
    raise ValueError(f"Unknown LLM provider: {provider!r}")
