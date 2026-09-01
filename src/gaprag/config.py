"""Central configuration, loaded from environment variables and an optional .env file.

Every setting is overridable with a ``GAPRAG_`` prefixed environment variable, e.g.
``GAPRAG_TOP_K=6``. Provider API keys use their conventional names
(``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``) and are read directly by the providers.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/gaprag/config.py -> repo root is two parents up from the package directory.
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent


class Settings(BaseSettings):
    """Application settings. Immutable per process (see :func:`get_settings`)."""

    model_config = SettingsConfigDict(
        env_prefix="GAPRAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Providers: which backend answers and which embeds.
    # LLM: ollama | openai | anthropic | bedrock | mock
    # Embedding: sentence_transformers | openai | bedrock | hash
    llm_provider: str = "ollama"
    embedding_provider: str = "sentence_transformers"

    # AWS region for the Bedrock provider (credentials use the standard AWS chain).
    aws_region: str = "us-east-1"

    # Retrieval backend: the native FAISS store, or a LangChain FAISS adapter.
    retriever: str = "native"  # native | langchain

    # Models
    llm_model: str = "llama3.1:8b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # Retrieval / chunking
    top_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Behaviour
    request_timeout: float = 60.0
    json_logs: bool = True

    # Paths (derived from the repo root by default)
    corpus_dir: Path = REPO_ROOT / "data" / "corpus"
    index_dir: Path = REPO_ROOT / "data" / "index"


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` instance."""
    return Settings()
