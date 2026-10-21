from collections.abc import Sequence


def precision_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """Fraction of the top-k retrieved documents that are relevant."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    top_k = list(retrieved[:k])
    if not top_k:
        return 0.0

    return sum(doc in relevant for doc in top_k) / len(top_k)


def recall_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """Fraction of all relevant documents retrieved in the top-k."""
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if not relevant:
        return 0.0

    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def mean_reciprocal_rank(
    retrieved: Sequence[str],
    relevant: set[str],
) -> float:
    """Reciprocal rank of the first relevant document."""
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank

    return 0.0
