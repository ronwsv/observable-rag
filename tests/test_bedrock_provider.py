"""Bedrock provider construction (no AWS calls). Skipped when boto3 is absent."""

import pytest

pytest.importorskip("boto3")

from gaprag.providers import build_embedding_provider, build_llm_provider  # noqa: E402


def test_bedrock_llm_builds_without_network():
    llm = build_llm_provider(
        "bedrock", "us.anthropic.claude-3-5-haiku-20241022-v1:0", region="us-east-1"
    )
    assert llm.name == "bedrock"
    assert "claude" in llm.model


def test_bedrock_embedding_reports_titan_dim():
    emb = build_embedding_provider("bedrock", "amazon.titan-embed-text-v2:0", region="us-east-1")
    assert emb.name == "bedrock"
    assert emb.dim == 1024
