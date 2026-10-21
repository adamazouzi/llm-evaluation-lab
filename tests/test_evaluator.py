import json

import pytest

from evaluation.evaluator import evaluate_dataset


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
    assert results["precision_at_k"] == pytest.approx(1 / 3)
    assert results["recall_at_k"] == pytest.approx(1.0)
    assert results["mrr"] == pytest.approx(0.75)


def test_evaluate_dataset_rejects_invalid_k(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError):
        evaluate_dataset(path, k=0)
