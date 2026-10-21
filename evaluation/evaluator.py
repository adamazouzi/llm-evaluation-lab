
import json
from pathlib import Path

from evaluation.metrics.retrieval import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def evaluate_dataset(dataset_path: str | Path, k: int = 3) -> dict:
    """Evaluate precomputed retrieval results from a JSON dataset."""
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
        reciprocal_ranks.append(
            mean_reciprocal_rank(retrieved, relevant)
        )

    count = len(dataset)

    return {
        "queries_evaluated": count,
        "k": k,
        "precision_at_k": sum(precisions) / count,
        "recall_at_k": sum(recalls) / count,
        "mrr": sum(reciprocal_ranks) / count,
    }


def evaluate_retriever(
    retriever,
    dataset: list[dict],
    *,
    k: int = 3,
) -> dict:
    """Evaluate a real retriever against manually annotated chunk IDs.

    Each dataset item must contain:
      - query: the search query
      - relevant_chunk_ids: IDs of relevant chunks

    Each search result must expose its ID through
    result.metadata["chunk_id"].
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty list")

    precisions = []
    recalls = []
    reciprocal_ranks = []

    for item in dataset:
        query = item["query"]
        relevant = set(item["relevant_chunk_ids"])

        results = retriever.search(query, top_k=k)

        retrieved = [
            result.metadata["chunk_id"]
            for result in results
            if "chunk_id" in result.metadata
        ]

        precisions.append(precision_at_k(retrieved, relevant, k))
        recalls.append(recall_at_k(retrieved, relevant, k))
        reciprocal_ranks.append(
            mean_reciprocal_rank(retrieved, relevant)
        )

    count = len(dataset)

    return {
        "queries_evaluated": count,
        "k": k,
        "precision_at_k": sum(precisions) / count,
        "recall_at_k": sum(recalls) / count,
        "mrr": sum(reciprocal_ranks) / count,
    }
