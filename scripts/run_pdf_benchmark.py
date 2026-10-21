"""Evaluate semantic retrieval on a PDF using manually annotated questions."""

import argparse
import json
from pathlib import Path

import pymupdf

from backend.app.services.document_loader import load_pdf_chunks
from backend.app.services.vector_search import VectorSearch
from evaluation.evaluator import evaluate_retriever

DEFAULT_PDF = Path("data/demo/pdf_benchmark_corpus.pdf")
DEFAULT_ANNOTATIONS = Path("data/demo/pdf_benchmark_questions.json")
DEFAULT_OUTPUT = Path("data/demo/pdf_benchmark_results.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEMO_PAGES = [
    (
        "Python",
        (
            "Python is a programming language used for automation, "
            "web development, and data analysis."
        ),
    ),
    (
        "Machine learning",
        (
            "Machine learning learns patterns from data to make predictions "
            "or decisions."
        ),
    ),
    (
        "Data engineering",
        (
            "Data engineering prepares reliable data and builds pipelines "
            "between systems."
        ),
    ),
    (
        "Deep learning",
        (
            "Deep learning uses neural networks with multiple layers to learn "
            "complex patterns."
        ),
    ),
    (
        "RAG",
        (
            "Retrieval augmented generation combines document retrieval with "
            "a language model to generate grounded answers."
        ),
    ),
]

DEMO_QUESTIONS = [
    {
        "query": "What can Python be used for?",
        "relevant_chunk_ids": ["pdf_benchmark_corpus.pdf::chunk-0"],
    },
    {
        "query": "How does machine learning learn patterns?",
        "relevant_chunk_ids": ["pdf_benchmark_corpus.pdf::chunk-1"],
    },
    {
        "query": "What is the purpose of data engineering?",
        "relevant_chunk_ids": ["pdf_benchmark_corpus.pdf::chunk-2"],
    },
    {
        "query": "How do neural networks learn complex patterns?",
        "relevant_chunk_ids": ["pdf_benchmark_corpus.pdf::chunk-3"],
    },
    {
        "query": "How does RAG generate grounded answers?",
        "relevant_chunk_ids": ["pdf_benchmark_corpus.pdf::chunk-4"],
    },
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate semantic retrieval on an annotated PDF."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        help="PDF to evaluate; defaults to the synthetic demo PDF.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="JSON file containing manually annotated questions.",
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
        help="Number of retrieved chunks to evaluate (default: 3).",
    )
    return parser.parse_args()


def create_demo_pdf(path: Path) -> None:
    """Create the synthetic PDF used by the default demonstration."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with pymupdf.open() as document:
        for title, content in DEMO_PAGES:
            page = document.new_page()
            page.insert_text(
                (72, 72),
                title,
                fontsize=14,
            )

            text_box = pymupdf.Rect(72, 105, 540, 760)
            remaining_space = page.insert_textbox(
                text_box,
                content,
                fontsize=12,
                lineheight=1.4,
            )

            if remaining_space < 0:
                raise ValueError(
                    f"Text does not fit on the demo PDF page: {title}"
                )

        document.save(path)

def load_questions(path: Path) -> list[dict]:
    """Load and validate annotated questions from JSON."""
    with path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list) or not questions:
        raise ValueError("Annotations must be a non-empty JSON list.")

    for index, item in enumerate(questions):
        if not isinstance(item, dict):
            raise TypeError(f"Annotation {index} must be a JSON object.")

        query = item.get("query")
        relevant_ids = item.get("relevant_chunk_ids")

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


def main() -> None:
    """Load a PDF, index its chunks, evaluate retrieval, and save results."""
    args = parse_args()

    if args.k < 1:
        raise ValueError("k must be at least 1.")

    using_demo = args.pdf is None
    pdf_path = args.pdf if args.pdf is not None else DEFAULT_PDF
    annotations_path = args.annotations

    if using_demo:
        create_demo_pdf(pdf_path)
        if annotations_path is None:
            annotations_path = DEFAULT_ANNOTATIONS
            annotations_path.parent.mkdir(parents=True, exist_ok=True)
            annotations_path.write_text(
                json.dumps(DEMO_QUESTIONS, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    elif annotations_path is None:
        raise ValueError(
            "For a custom PDF, provide --annotations with questions "
            "and relevant chunk IDs for that document."
        )

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The input file must have a .pdf extension.")

    if annotations_path is None or not annotations_path.is_file():
        raise FileNotFoundError(
            "Annotations JSON not found. Provide a valid --annotations path."
        )

    questions = load_questions(annotations_path)

    print(f"Loading PDF: {pdf_path}")
    chunks = load_pdf_chunks(pdf_path)

    if not chunks:
        raise ValueError("The PDF did not produce any text chunks.")

    texts = [chunk.text for chunk in chunks]
    metadata = [
        {
            "chunk_id": f"{chunk.document_name}::chunk-{chunk.chunk_index}",
            "document_name": chunk.document_name,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunks
    ]

    actual_ids = {item["chunk_id"] for item in metadata}
    for question in questions:
        unknown_ids = set(question["relevant_chunk_ids"]) - actual_ids
        if unknown_ids:
            raise ValueError(
                f"Unknown relevant chunk IDs for query {question['query']!r}: "
                f"{sorted(unknown_ids)}. Check the annotations for this PDF."
            )

    print(f"Extracted {len(chunks)} chunks.")
    print(f"Loading embedding model: {MODEL_NAME}")

    retriever = VectorSearch(model_name=MODEL_NAME)
    retriever.index(texts, metadata)

    metrics = evaluate_retriever(retriever, questions, k=args.k)
    query_results = []

    for question in questions:
        results = retriever.search(question["query"], top_k=args.k)
        relevant_ids = set(question["relevant_chunk_ids"])

        query_results.append(
            {
                "query": question["query"],
                "relevant_chunk_ids": question["relevant_chunk_ids"],
                "retrieved_results": [
                    {
                        "rank": rank,
                        "chunk_id": result.metadata["chunk_id"],
                        "document_name": result.metadata["document_name"],
                        "page_number": result.metadata["page_number"],
                        "score": round(result.score, 4),
                        "is_relevant": result.metadata["chunk_id"] in relevant_ids,
                        "text": result.text,
                    }
                    for rank, result in enumerate(results, start=1)
                ],
            }
        )

    output = {
        "benchmark_type": (
            "synthetic_pdf_retrieval_demo"
            if using_demo
            else "pdf_retrieval_benchmark"
        ),
        "corpus_pdf": str(pdf_path),
        "annotations_file": str(annotations_path),
        "model": MODEL_NAME,
        "corpus_chunks": len(chunks),
        **metrics,
        "query_results": query_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nPDF benchmark summary")
    print("---------------------")
    print(f"Queries evaluated: {metrics['queries_evaluated']}")
    print(f"Corpus chunks: {len(chunks)}")
    print(f"Precision@{args.k}: {metrics['precision_at_k']:.3f}")
    print(f"Recall@{args.k}: {metrics['recall_at_k']:.3f}")
    print(f"MRR: {metrics['mrr']:.3f}")
    print(f"Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()