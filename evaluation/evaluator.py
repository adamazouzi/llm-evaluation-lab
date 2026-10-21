import json
from pathlib import Path

from evaluation.metrics.retrieval import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def evaluate_dataset(dataset_path: str | Path, k: int = 3) -> dict:
    """Evaluate retrieved documents against known relevant documents."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    path = Path(dataset_path)
    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty JSON list")

    precisions = []
    recalls = []
    reciprocal_ranks = []

    for item in dataset:
        relevant = set(item["relevant_documents"])
        retrieved = item["retrieved_documents"]

        precisions.append(precision_at_k(retrieved, relevant, k))
        recalls.append(recall_at_k(retrieved, relevant, k))
        reciprocal_ranks.append(mean_reciprocal_rank(retrieved, relevant))

    count = len(dataset)

    return {
        "queries_evaluated": count,
        "k": k,
        "precision_at_k": sum(precisions) / count,
        "recall_at_k": sum(recalls) / count,
        "mrr": sum(reciprocal_ranks) / count,
    }
