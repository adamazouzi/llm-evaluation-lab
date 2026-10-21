
import json
from types import SimpleNamespace

import pytest

from evaluation.evaluator import evaluate_dataset, evaluate_retriever


def test_evaluate_dataset(tmp_path):
    dataset = [
        {
            "query": "Question 1",
            "relevant_documents": ["a"],
            "retrieved_documents": ["a", "b", "c"],
        },
        {
            "query": "Question 2",
            "relevant_documents": ["b"],
            "retrieved_documents": ["c", "b", "a"],
        },
    ]

    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    results = evaluate_dataset(path, k=3)

    assert results["queries_evaluated"] == 2
    assert results["k"] == 3
    assert results["precision_at_k"] == pytest.approx(1 / 3)
    assert results["recall_at_k"] == pytest.approx(1.0)
    assert results["mrr"] == pytest.approx(0.75)


def test_evaluate_dataset_rejects_invalid_k(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="k must be greater than 0"):
        evaluate_dataset(path, k=0)


def test_evaluate_retriever_with_search_results():
    """Check metrics computed from results returned by a retriever."""

    class FakeRetriever:
        def search(self, query: str, top_k: int = 3):
            results = {
                "What is Python?": [
                    SimpleNamespace(
                        metadata={"chunk_id": "python"},
                        text="Python is a programming language.",
                        score=0.95,
                    ),
                    SimpleNamespace(
                        metadata={"chunk_id": "ml"},
                        text="Machine learning uses data.",
                        score=0.70,
                    ),
                ]
            }
            return results[query][:top_k]

    dataset = [
        {
            "query": "What is Python?",
            "relevant_chunk_ids": ["python"],
        }
    ]

    results = evaluate_retriever(FakeRetriever(), dataset, k=2)

    assert results["queries_evaluated"] == 1
    assert results["k"] == 2
    assert results["precision_at_k"] == pytest.approx(0.5)
    assert results["recall_at_k"] == pytest.approx(1.0)
    assert results["mrr"] == pytest.approx(1.0)


def test_evaluate_retriever_rejects_invalid_k():
    class FakeRetriever:
        def search(self, query: str, top_k: int = 3):
            return []

    dataset = [
        {
            "query": "What is Python?",
            "relevant_chunk_ids": ["python"],
        }
    ]

    with pytest.raises(ValueError, match="k must be greater than 0"):
        evaluate_retriever(FakeRetriever(), dataset, k=0)
