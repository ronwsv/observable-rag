"""Gemini provider construction (no API calls). Skipped when google-genai is absent."""

import pytest

pytest.importorskip("google.genai")

from gaprag.providers import build_embedding_provider, build_llm_provider  # noqa: E402


def test_gemini_llm_builds_without_key():
    llm = build_llm_provider("gemini", "gemini-2.0-flash")
    assert llm.name == "gemini"
    assert llm.model == "gemini-2.0-flash"


def test_gemini_embedding_reports_dim():
    emb = build_embedding_provider("gemini", "text-embedding-004")
    assert emb.name == "gemini"
    assert emb.dim == 768
