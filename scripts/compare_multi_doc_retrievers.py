
"""Compare lexical and semantic retrieval on a multi-document PDF corpus."""

import argparse
import json
import statistics
import time
from pathlib import Path

from backend.app.services.document_loader import load_pdf_chunks
from backend.app.services.keyword_search import TfidfSearch
from backend.app.services.vector_search import VectorSearch
from evaluation.metrics.retrieval import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)

DEFAULT_CORPUS_DIR = Path("data/demo/multi_doc_corpus")
DEFAULT_ANNOTATIONS = Path("data/demo/multi_doc_questions.json")
DEFAULT_OUTPUT = Path("data/demo/multi_doc_retriever_results.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare TF-IDF and semantic retrieval on multiple PDF documents."
        )
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help=f"Directory containing PDF documents (default: {DEFAULT_CORPUS_DIR}).",
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
    """Load and validate annotated evaluation questions."""
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
    """Evaluate retrieval quality and performance while retaining details."""
    precisions = []
    recalls = []
    reciprocal_ranks = []
    search_times = []
    query_results = []

    for question in questions:
        relevant_ids = set(question["relevant_chunk_ids"])

        start_time = time.perf_counter()
        results = retriever.search(question["query"], top_k=k)
        search_duration = time.perf_counter() - start_time

        search_times.append(search_duration)

        retrieved_ids = [
            result.metadata["chunk_id"]
            for result in results
            if "chunk_id" in result.metadata
        ]

        precisions.append(
            precision_at_k(retrieved_ids, relevant_ids, k)
        )
        recalls.append(
            recall_at_k(retrieved_ids, relevant_ids, k)
        )
        reciprocal_ranks.append(
            mean_reciprocal_rank(retrieved_ids, relevant_ids)
        )

        query_results.append(
            {
                "query": question["query"],
                "relevant_chunk_ids": sorted(relevant_ids),
                "retrieved_chunks": [
                    {
                        "rank": rank,
                        "chunk_id": result.metadata["chunk_id"],
                        "document_name": result.metadata.get(
                            "document_name"
                        ),
                        "page_number": result.metadata.get("page_number"),
                        "score": round(float(result.score), 6),
                        "is_relevant": (
                            result.metadata["chunk_id"] in relevant_ids
                        ),
                    }
                    for rank, result in enumerate(results, start=1)
                ],
                "search_time_ms": round(search_duration * 1000, 4),
            }
        )

    count = len(questions)
    total_search_time = sum(search_times)

    return {
        "metrics": {
            "queries_evaluated": count,
            "k": k,
            "precision_at_k": sum(precisions) / count,
            "recall_at_k": sum(recalls) / count,
            "mrr": sum(reciprocal_ranks) / count,
        },
        "performance": {
            "total_search_time_seconds": round(total_search_time, 6),
            "average_search_time_ms": round(
                statistics.mean(search_times) * 1000,
                4,
            ),
            "median_search_time_ms": round(
                statistics.median(search_times) * 1000,
                4,
            ),
            "queries_per_second": round(
                count / total_search_time,
                4,
            ) if total_search_time > 0 else None,
        },
        "query_results": query_results,
    }


def main() -> None:
    """Run the retrieval comparison across all PDFs in the corpus."""
    args = parse_args()

    if args.k < 1:
        raise ValueError("k must be at least 1.")

    if not args.corpus_dir.is_dir():
        raise FileNotFoundError(
            f"Corpus directory not found: {args.corpus_dir}"
        )

    pdf_paths = sorted(
        path
        for path in args.corpus_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_paths:
        raise ValueError(
            f"No PDF documents found in: {args.corpus_dir}"
        )

    questions = load_questions(args.annotations)
    chunks = []
    metadata = []

    for pdf_path in pdf_paths:
        document_chunks = load_pdf_chunks(pdf_path)

        for chunk in document_chunks:
            chunks.append(chunk.text)
            metadata.append(
                {
                    "chunk_id": (
                        f"{chunk.document_name}::chunk-{chunk.chunk_index}"
                    ),
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                }
            )

    if not chunks:
        raise ValueError("The PDF corpus did not produce any text chunks.")

    actual_ids = {item["chunk_id"] for item in metadata}

    for index, question in enumerate(questions):
        unknown_ids = set(question["relevant_chunk_ids"]) - actual_ids
        if unknown_ids:
            raise ValueError(
                f"Unknown relevant chunk IDs in annotation {index} "
                f"({question['query']!r}): {sorted(unknown_ids)}. "
                "Check the annotations against the loaded corpus."
            )

    print(f"Corpus directory: {args.corpus_dir}")
    print(f"PDF documents: {len(pdf_paths)}")
    for pdf_path in pdf_paths:
        print(f"  - {pdf_path.name}")
    print(f"Corpus chunks: {len(chunks)}")
    print(f"Queries: {len(questions)}")
    print(f"Evaluating with k={args.k}")

    print("\nRunning TF-IDF retrieval...")
    lexical_retriever = TfidfSearch()

    indexing_start = time.perf_counter()
    lexical_retriever.index(chunks, metadata)
    lexical_indexing_time = time.perf_counter() - indexing_start

    lexical_results = evaluate_with_details(
        lexical_retriever,
        questions,
        k=args.k,
    )

    print(f"\nLoading embedding model: {MODEL_NAME}")
    semantic_retriever = VectorSearch(model_name=MODEL_NAME)

    indexing_start = time.perf_counter()
    semantic_retriever.index(chunks, metadata)
    semantic_indexing_time = time.perf_counter() - indexing_start

    semantic_results = evaluate_with_details(
        semantic_retriever,
        questions,
        k=args.k,
    )

    output = {
        "benchmark_type": "multi_document_retriever_comparison",
        "corpus_directory": str(args.corpus_dir),
        "corpus_documents": [path.name for path in pdf_paths],
        "annotations_file": str(args.annotations),
        "corpus_chunks": len(chunks),
        "queries_evaluated": len(questions),
        "k": args.k,
        "methods": {
            "tfidf": {
                "description": "Lexical retrieval using TF-IDF",
                "indexing_time_seconds": round(
                    lexical_indexing_time,
                    6,
                ),
                **lexical_results,
            },
            "semantic": {
                "description": f"Embedding retrieval using {MODEL_NAME}",
                "model": MODEL_NAME,
                "indexing_time_seconds": round(
                    semantic_indexing_time,
                    6,
                ),
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
    print(f"{'Metric':<28}{'TF-IDF':>14}{'Semantic':>14}")

    metric_rows = (
        ("Precision@K", "precision_at_k"),
        ("Recall@K", "recall_at_k"),
        ("MRR", "mrr"),
    )

    for label, metric in metric_rows:
        lexical_value = lexical_results["metrics"][metric]
        semantic_value = semantic_results["metrics"][metric]

        print(
            f"{label:<28}{lexical_value:>14.3f}"
            f"{semantic_value:>14.3f}"
        )

    print("\nPerformance comparison")
    print("----------------------")

    performance_rows = (
        ("Indexing time (seconds)", "indexing_time_seconds"),
    )

    for label, metric in performance_rows:
        lexical_value = output["methods"]["tfidf"][metric]
        semantic_value = output["methods"]["semantic"][metric]

        print(
            f"{label:<28}{lexical_value:>14.4f}"
            f"{semantic_value:>14.4f}"
        )

    for label, metric in (
        ("Total search time (seconds)", "total_search_time_seconds"),
        ("Average search time (ms)", "average_search_time_ms"),
        ("Median search time (ms)", "median_search_time_ms"),
        ("Queries per second", "queries_per_second"),
    ):
        lexical_value = lexical_results["performance"][metric]
        semantic_value = semantic_results["performance"][metric]

        print(
            f"{label:<28}{lexical_value:>14.4f}"
            f"{semantic_value:>14.4f}"
        )

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
