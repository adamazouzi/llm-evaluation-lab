
"""Tests for the multi-document retrieval benchmark."""

from types import SimpleNamespace

import pytest

from scripts.compare_multi_doc_retrievers import evaluate_with_details


class FakeRetriever:
    """Return deterministic search results for benchmark tests."""

    def search(self, query: str, top_k: int = 3):
        results = {
            "Question 1": [
                SimpleNamespace(
                    metadata={
                        "chunk_id": "chunk-a",
                        "document_name": "python_guide.pdf",
                        "page_number": 1,
                    },
                    text="Python is a programming language.",
                    score=0.95,
                ),
                SimpleNamespace(
                    metadata={
                        "chunk_id": "chunk-b",
                        "document_name": "ml_guide.pdf",
                        "page_number": 2,
                    },
                    text="Machine learning uses data.",
                    score=0.70,
                ),
            ],
            "Question 2": [
                SimpleNamespace(
                    metadata={
                        "chunk_id": "chunk-c",
                        "document_name": "rag_guide.pdf",
                        "page_number": 1,
                    },
                    text="RAG combines retrieval and generation.",
                    score=0.90,
                ),
                SimpleNamespace(
                    metadata={
                        "chunk_id": "chunk-d",
                        "document_name": "rag_guide.pdf",
                        "page_number": 2,
                    },
                    text="Retrieval finds relevant passages.",
                    score=0.80,
                ),
            ],
        }

        return results[query][:top_k]


def test_evaluate_with_details_calculates_metrics_and_performance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check retrieval metrics and timing calculations."""
    timestamps = iter([10.0, 10.002, 20.0, 20.004])

    monkeypatch.setattr(
        "scripts.compare_multi_doc_retrievers.time.perf_counter",
        lambda: next(timestamps),
    )

    questions = [
        {
            "query": "Question 1",
            "relevant_chunk_ids": ["chunk-a"],
        },
        {
            "query": "Question 2",
            "relevant_chunk_ids": ["chunk-d"],
        },
    ]

    results = evaluate_with_details(
        FakeRetriever(),
        questions,
        k=2,
    )

    assert results["metrics"]["queries_evaluated"] == 2
    assert results["metrics"]["k"] == 2
    assert results["metrics"]["precision_at_k"] == pytest.approx(0.5)
    assert results["metrics"]["recall_at_k"] == pytest.approx(1.0)
    assert results["metrics"]["mrr"] == pytest.approx(0.75)

    performance = results["performance"]

    assert performance["total_search_time_seconds"] == pytest.approx(0.006)
    assert performance["average_search_time_ms"] == pytest.approx(3.0)
    assert performance["median_search_time_ms"] == pytest.approx(3.0)
    assert performance["queries_per_second"] == pytest.approx(
        2 / 0.006
    )

    assert len(results["query_results"]) == 2
    assert results["query_results"][0]["search_time_ms"] == pytest.approx(
        2.0
    )
    assert results["query_results"][1]["search_time_ms"] == pytest.approx(
        4.0
    )


def test_evaluate_with_details_preserves_retrieved_chunk_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that detailed results retain document and page metadata."""
    timestamps = iter([1.0, 1.001])

    monkeypatch.setattr(
        "scripts.compare_multi_doc_retrievers.time.perf_counter",
        lambda: next(timestamps),
    )

    questions = [
        {
            "query": "Question 1",
            "relevant_chunk_ids": ["chunk-a"],
        }
    ]

    results = evaluate_with_details(
        FakeRetriever(),
        questions,
        k=2,
    )

    first_chunk = results["query_results"][0]["retrieved_chunks"][0]

    assert first_chunk["chunk_id"] == "chunk-a"
    assert first_chunk["document_name"] == "python_guide.pdf"
    assert first_chunk["page_number"] == 1
    assert first_chunk["rank"] == 1
    assert first_chunk["is_relevant"] is True
    assert first_chunk["score"] == pytest.approx(0.95)
