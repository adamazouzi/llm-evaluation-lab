
"""Run a semantic retrieval benchmark on a small demo corpus."""

import json
from pathlib import Path

from backend.app.services.vector_search import VectorSearch
from evaluation.evaluator import evaluate_retriever


def main() -> None:
    """Index a demo corpus, evaluate retrieval, and save detailed results."""
    chunks = [
        (
            "python",
            (
                "Python is a programming language widely used for automation, "
                "web development, and data analysis."
            ),
        ),
        (
            "machine-learning",
            (
                "Machine learning is a branch of artificial intelligence "
                "that learns patterns from data."
            ),
        ),
        (
            "data-engineering",
            (
                "Data engineering focuses on collecting, transforming, "
                "and preparing reliable data pipelines."
            ),
        ),
        (
            "deep-learning",
            (
                "Deep learning uses neural networks with multiple layers "
                "to learn complex patterns."
            ),
        ),
        (
            "rag",
            (
                "Retrieval augmented generation combines document retrieval "
                "with a language model to generate grounded answers."
            ),
        ),
    ]

    questions = [
        {
            "query": "What is Python used for?",
            "relevant_chunk_ids": ["python"],
        },
        {
            "query": "How does machine learning work?",
            "relevant_chunk_ids": ["machine-learning"],
        },
        {
            "query": "What is data engineering?",
            "relevant_chunk_ids": ["data-engineering"],
        },
        {
            "query": "What are deep neural networks?",
            "relevant_chunk_ids": ["deep-learning"],
        },
        {
            "query": "What is retrieval augmented generation?",
            "relevant_chunk_ids": ["rag"],
        },
    ]

    retriever = VectorSearch()

    texts = [text for _, text in chunks]
    metadata = [{"chunk_id": chunk_id} for chunk_id, _ in chunks]
    text_by_id = {chunk_id: text for chunk_id, text in chunks}

    print("Indexing demo corpus with sentence embeddings...")
    retriever.index(texts, metadata)

    print("Evaluating retrieval...")
    metrics = evaluate_retriever(retriever, questions, k=3)

    query_results = []

    for item in questions:
        results = retriever.search(item["query"], top_k=3)
        relevant_ids = set(item["relevant_chunk_ids"])
        retrieved_ids = [
            result.metadata["chunk_id"] for result in results
        ]

        query_results.append(
            {
                "query": item["query"],
                "relevant_chunk_ids": item["relevant_chunk_ids"],
                "retrieved_results": [
                    {
                        "rank": rank,
                        "chunk_id": result.metadata["chunk_id"],
                        "score": round(result.score, 4),
                        "is_relevant": (
                            result.metadata["chunk_id"] in relevant_ids
                        ),
                        "text": text_by_id[result.metadata["chunk_id"]],
                    }
                    for rank, result in enumerate(results, start=1)
                ],
                "first_result_is_relevant": (
                    bool(retrieved_ids)
                    and retrieved_ids[0] in relevant_ids
                ),
            }
        )

    output = {
        "benchmark_type": "real_embedding_demo",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "corpus_chunks": len(chunks),
        **metrics,
        "query_results": query_results,
    }

    output_path = Path("data/demo/retrieval_benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nBenchmark summary")
    print("-----------------")
    print(f"Queries evaluated: {metrics['queries_evaluated']}")
    print(f"Precision@{metrics['k']}: {metrics['precision_at_k']:.3f}")
    print(f"Recall@{metrics['k']}:    {metrics['recall_at_k']:.3f}")
    print(f"MRR:          {metrics['mrr']:.3f}")

    print("\nResults by query")
    print("----------------")
    for item in query_results:
        print(f"\nQuestion: {item['query']}")
        print(f"Expected: {', '.join(item['relevant_chunk_ids'])}")

        for result in item["retrieved_results"]:
            status = "relevant" if result["is_relevant"] else "not relevant"
            print(
                f"  {result['rank']}. {result['chunk_id']} "
                f"(score={result['score']:.4f}, {status})"
            )

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
