
"""Tests for the RAG API endpoint."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_ask_returns_answer_and_sources() -> None:
    response = client.post(
        "/ask",
        json={"question": "What is Python used for?"},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["question"] == "What is Python used for?"
    assert body["answer"]
    assert isinstance(body["sources"], list)
    assert body["latency_ms"] >= 0
    assert isinstance(body["abstained"], bool)


def test_ask_rejects_empty_question() -> None:
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_rejects_missing_question() -> None:
    response = client.post("/ask", json={})

    assert response.status_code == 422


def test_health_endpoint_still_works() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
