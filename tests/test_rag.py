def test_pipeline_answers_with_citations_and_trace(pipeline):
    resp = pipeline.answer("What is Retrieval-Augmented Generation?")
    assert resp.route == "retrieve"
    assert resp.answer
    assert resp.citations, "a retrieval answer should carry citations"
    assert resp.trace is not None
    assert resp.trace.total_tokens > 0
    assert resp.trace.retrieved_ids


def test_pipeline_direct_route_has_no_citations(pipeline):
    resp = pipeline.answer("hello")
    assert resp.route == "direct"
    assert resp.citations == []


def test_trace_records_latency(pipeline):
    resp = pipeline.answer("What is FAISS?")
    assert resp.trace.total_ms >= 0.0
    assert resp.trace.model
