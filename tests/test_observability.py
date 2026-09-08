from gaprag.observability import RequestTrace, estimate_cost_usd


def test_unknown_or_local_model_is_free():
    assert estimate_cost_usd("llama3.1:8b", 1000, 1000) == 0.0


def test_longest_key_wins_for_overlapping_names():
    # "gpt-4o" is a substring of "gpt-4o-mini"; the longer, specific key must win.
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 0) == 0.15


def test_bedrock_model_id_resolves_to_price():
    # A Bedrock id should map to the same price as the direct provider id.
    assert estimate_cost_usd("us.anthropic.claude-3-5-haiku-20241022-v1:0", 1_000_000, 0) == 0.80


def test_gemini_model_id_resolves_to_price():
    assert estimate_cost_usd("gemini-2.0-flash", 1_000_000, 0) == 0.10


def test_trace_finalize_computes_cost():
    trace = RequestTrace(
        query="q", model="gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=0
    )
    trace.finalize()
    assert trace.cost_usd == 0.15
