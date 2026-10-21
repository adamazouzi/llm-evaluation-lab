import pytest

from evaluation.metrics.retrieval import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5


def test_mrr():
    assert mean_reciprocal_rank(["x", "b", "a"], {"a", "b"}) == 0.5


def test_no_relevant_documents():
    assert precision_at_k(["a"], set(), 1) == 0.0
    assert recall_at_k(["a"], set(), 1) == 0.0
    assert mean_reciprocal_rank(["a"], set()) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, 0)
