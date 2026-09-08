"""Provider registry: embeddings and LLMs behind a common interface."""

from .base import EmbeddingProvider, LLMProvider, LLMResult, l2_normalize
from .embeddings import (
    BedrockEmbedding,
    GeminiEmbedding,
    HashEmbedding,
    OpenAIEmbedding,
    SentenceTransformerEmbedding,
    build_embedding_provider,
)
from .llm import (
    AnthropicLLM,
    BedrockLLM,
    GeminiLLM,
    MockLLM,
    OllamaLLM,
    OpenAILLM,
    build_llm_provider,
)

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "LLMResult",
    "l2_normalize",
    "HashEmbedding",
    "SentenceTransformerEmbedding",
    "OpenAIEmbedding",
    "BedrockEmbedding",
    "GeminiEmbedding",
    "build_embedding_provider",
    "MockLLM",
    "OllamaLLM",
    "OpenAILLM",
    "AnthropicLLM",
    "BedrockLLM",
    "GeminiLLM",
    "build_llm_provider",
]
