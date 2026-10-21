import json

from evaluation.evaluator import evaluate_dataset


def main() -> None:
    results = evaluate_dataset(
        "data/demo/evaluation_dataset.json",
        k=3,
    )

    print("\n=== LLM Evaluation Lab ===")
    print(f"Queries evaluated: {results['queries_evaluated']}")
    print(f"Precision@{results['k']}: {results['precision_at_k']:.3f}")
    print(f"Recall@{results['k']}:    {results['recall_at_k']:.3f}")
    print(f"MRR:             {results['mrr']:.3f}")

    with open("data/demo/evaluation_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nResults saved to data/demo/evaluation_results.json")


if __name__ == "__main__":
    main()
