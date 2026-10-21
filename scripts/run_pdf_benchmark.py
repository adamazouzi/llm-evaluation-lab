"""Run a semantic retrieval benchmark using the PDF ingestion pipeline."""

import json
from pathlib import Path

import pymupdf

from backend.app.services.document_loader import load_pdf_chunks
from backend.app.services.vector_search import VectorSearch
from evaluation.evaluator import evaluate_retriever

PDF_PATH = Path("data/demo/pdf_benchmark_corpus.pdf")
OUTPUT_PATH = Path("data/demo/pdf_benchmark_results.json")
TOP_K = 3

DEMO_PAGES = [
    ("Python", "Python is a programming language used for automation, web development, and data analysis."),
    ("Machine learning", "Machine learning learns patterns from data to make predictions or decisions."),
    ("Data engineering", "Data engineering prepares reliable data and builds pipelines between systems."),
    ("Deep learning", "Deep learning uses neural networks with multiple layers to learn complex patterns."),
    ("RAG", "Retrieval augmented generation combines document retrieval with a language model to generate grounded answers."),
]


def create_demo_pdf(path: Path) -> None:
    """Create a small synthetic PDF for testing the ingestion pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with pymupdf.open() as document:
        for title, content in DEMO_PAGES:
            page = document.new_page()
            page.insert_text((72, 72), f"{title}\n\n{content}", fontsize=12)
        document.save(path)


def main() -> None:
    """Load a PDF, index its chunks, and evaluate retrieval."""
    create_demo_pdf(PDF_PATH)

    print(f"Loading PDF: {PDF_PATH}")
    chunks = load_pdf_chunks(PDF_PATH)

    if len(chunks) != len(DEMO_PAGES):
        raise RuntimeError(
            f"Expected {len(DEMO_PAGES)} chunks, but loaded {len(chunks)}."
        )

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

    questions = [
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

    actual_ids = {item["chunk_id"] for item in metadata}
    for question in questions:
        if not set(question["relevant_chunk_ids"]).issubset(actual_ids):
            raise ValueError("A relevant chunk ID does not exist in the PDF.")

    retriever = VectorSearch()
    retriever.index(texts, metadata)

    metrics = evaluate_retriever(retriever, questions, k=TOP_K)

    query_results = []
    for question in questions:
        results = retriever.search(question["query"], top_k=TOP_K)
        relevant_ids = set(question["relevant_chunk_ids"])

        query_results.append(
            {
                "query": question["query"],
                "relevant_chunk_ids": question["relevant_chunk_ids"],
                "retrieved_results": [
                    {
                        "rank": rank,
                        "chunk_id": result.metadata["chunk_id"],
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
        "benchmark_type": "synthetic_pdf_retrieval_demo",
        "corpus_pdf": str(PDF_PATH),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "corpus_chunks": len(chunks),
        **metrics,
        "query_results": query_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nPDF benchmark summary")
    print("---------------------")
    print(f"Queries evaluated: {metrics['queries_evaluated']}")
    print(f"Corpus chunks: {len(chunks)}")
    print(f"Precision@{TOP_K}: {metrics['precision_at_k']:.3f}")
    print(f"Recall@{TOP_K}: {metrics['recall_at_k']:.3f}")
    print(f"MRR: {metrics['mrr']:.3f}")
    print(f"\nPDF created: {PDF_PATH}")
    print(f"Detailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()