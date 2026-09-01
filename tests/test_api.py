from fastapi.testclient import TestClient

from gaprag import api


def test_health_ok():
    client = TestClient(api.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_endpoint(pipeline):
    api._pipeline = pipeline  # inject the offline test pipeline
    try:
        client = TestClient(api.app)
        resp = client.post("/chat", json={"query": "What is chunk overlap?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"]
        assert data["route"] in {"retrieve", "direct"}
        assert "trace" in data and data["trace"]["total_tokens"] >= 0
    finally:
        api._pipeline = None
