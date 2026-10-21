"""Compare lexical TF-IDF retrieval with semantic embedding retrieval."""

import argparse
import json
from pathlib import Path

from backend.app.services.document_loader import load_pdf_chunks
from backend.app.services.keyword_search import TfidfSearch
from backend.app.services.vector_search import VectorSearch
from evaluation.evaluator import evaluate_retriever

DEFAULT_PDF = Path("data/demo/pdf_benchmark_corpus.pdf")
DEFAULT_ANNOTATIONS = Path("data/demo/pdf_benchmark_questions.json")
DEFAULT_OUTPUT = Path("data/demo/retriever_comparison_results.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF and semantic retrieval on the same PDF."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=DEFAULT_PDF,
        help="PDF to evaluate (default: demo PDF).",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="JSON file containing annotated questions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=3,
        help="Number of retrieved chunks per query (default: 3).",
    )
    return parser.parse_args()


def load_questions(path: Path) -> list[dict]:
    """Load and validate the annotated evaluation questions."""
    if not path.is_file():
        raise FileNotFoundError(f"Annotations file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list) or not questions:
        raise ValueError("Annotations must be a non-empty JSON list.")

    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise TypeError(f"Annotation {index} must be a JSON object.")

        query = question.get("query")
        relevant_ids = question.get("relevant_chunk_ids")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Annotation {index} needs a non-empty query."
            )

        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
            or any(
                not isinstance(chunk_id, str) or not chunk_id.strip()
                for chunk_id in relevant_ids
            )
        ):
            raise ValueError(
                f"Annotation {index} needs a non-empty list of chunk IDs."
            )

    return questions


def evaluate_with_details(
    retriever,
    questions: list[dict],
    *,
    k: int,
) -> dict:
    """Evaluate a retriever and retain its per-query retrieval details."""
    metrics = evaluate_retriever(retriever, questions, k=k)
    query_results = []

    for question in questions:
        relevant_ids = set(question["relevant_chunk_ids"])
        results = retriever.search(question["query"], top_k=k)

        query_results.append(
            {
                "query": question["query"],
                "relevant_chunk_ids": sorted(relevant_ids),
                "retrieved_chunks": [
                    {
                        "rank": rank,
                        "chunk_id": result.metadata["chunk_id"],
                        "document_name": result.metadata.get("document_name"),
                        "page_number": result.metadata.get("page_number"),
                        "score": round(result.score, 6),
                        "is_relevant": (
                            result.metadata["chunk_id"] in relevant_ids
                        ),
                    }
                    for rank, result in enumerate(results, start=1)
                ],
            }
        )

    return {
        "metrics": metrics,
        "query_results": query_results,
    }


def main() -> None:
    """Run both retrieval methods on the same PDF and questions."""
    args = parse_args()

    if args.k < 1:
        raise ValueError("k must be at least 1.")

    if not args.pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    if args.pdf.suffix.lower() != ".pdf":
        raise ValueError("The input file must have a .pdf extension.")

    questions = load_questions(args.annotations)
    chunks = load_pdf_chunks(args.pdf)

    if not chunks:
        raise ValueError("The PDF did not produce any text chunks.")

    texts = [chunk.text for chunk in chunks]
    metadata = [
        {
            "chunk_id": (
                f"{chunk.document_name}::chunk-{chunk.chunk_index}"
            ),
            "document_name": chunk.document_name,
            "page_number": chunk.page_number,
        }
        for chunk in chunks
    ]

    actual_ids = {item["chunk_id"] for item in metadata}

    for question in questions:
        unknown_ids = set(question["relevant_chunk_ids"]) - actual_ids
        if unknown_ids:
            raise ValueError(
                f"Unknown relevant chunk IDs for query {question['query']!r}: "
                f"{sorted(unknown_ids)}. Check the annotations."
            )

    print(f"Loading PDF: {args.pdf}")
    print(f"Corpus chunks: {len(chunks)}")
    print(f"Queries: {len(questions)}")
    print(f"Evaluating with k={args.k}")

    print("\nRunning TF-IDF retrieval...")
    lexical_retriever = TfidfSearch()
    lexical_retriever.index(texts, metadata)
    lexical_results = evaluate_with_details(
        lexical_retriever,
        questions,
        k=args.k,
    )

    print(f"\nLoading embedding model: {MODEL_NAME}")
    semantic_retriever = VectorSearch(model_name=MODEL_NAME)
    semantic_retriever.index(texts, metadata)
    semantic_results = evaluate_with_details(
        semantic_retriever,
        questions,
        k=args.k,
    )

    output = {
        "benchmark_type": "retriever_comparison",
        "corpus_pdf": str(args.pdf),
        "annotations_file": str(args.annotations),
        "corpus_chunks": len(chunks),
        "queries_evaluated": len(questions),
        "k": args.k,
        "methods": {
            "tfidf": {
                "description": "Lexical retrieval using TF-IDF",
                **lexical_results,
            },
            "semantic": {
                "description": f"Embedding retrieval using {MODEL_NAME}",
                "model": MODEL_NAME,
                **semantic_results,
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nRetrieval comparison")
    print("--------------------")
    print(f"{'Metric':<18}{'TF-IDF':>12}{'Semantic':>12}")

    for metric, label in (
        ("precision_at_k", f"Precision@{args.k}"),
        ("recall_at_k", f"Recall@{args.k}"),
        ("mrr", "MRR"),
    ):
        lexical_value = lexical_results["metrics"][metric]
        semantic_value = semantic_results["metrics"][metric]
        print(
            f"{label:<18}{lexical_value:>12.3f}{semantic_value:>12.3f}"
        )

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
