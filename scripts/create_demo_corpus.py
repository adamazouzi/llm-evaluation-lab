"""Generate a multi-document PDF corpus for retrieval experiments."""

from pathlib import Path

import pymupdf

OUTPUT_DIR = Path("data/demo/multi_doc_corpus")

DOCUMENTS = {
    "python_guide.pdf": [
        (
            "Python for automation",
            (
                "Python is widely used to automate repetitive tasks. "
                "Scripts can rename files, process spreadsheets, call APIs, "
                "and transform large collections of documents."
            ),
        ),
        (
            "Python for web development",
            (
                "Python supports web applications through frameworks such as "
                "Django and Flask. Applications can expose endpoints, validate "
                "requests, and communicate with databases."
            ),
        ),
        (
            "Python for data analysis",
            (
                "Data analysts use Python to clean datasets, calculate statistics, "
                "and create visualizations. Libraries such as pandas and NumPy "
                "help process structured numerical information."
            ),
        ),
    ],
    "machine_learning_guide.pdf": [
        (
            "Supervised learning",
            (
                "Supervised machine learning uses labeled examples. A model learns "
                "relationships between input features and target values, then "
                "predicts outcomes for previously unseen examples."
            ),
        ),
        (
            "Model evaluation",
            (
                "A machine learning model should be evaluated on data that was "
                "not used for training. Precision, recall, and F1 score measure "
                "different aspects of classification performance."
            ),
        ),
        (
            "Overfitting",
            (
                "Overfitting occurs when a model learns details specific to its "
                "training examples and performs poorly on new data. Regularization "
                "and cross-validation can help identify or reduce this problem."
            ),
        ),
    ],
    "data_engineering_guide.pdf": [
        (
            "Data pipelines",
            (
                "Data engineering builds pipelines that move information between "
                "systems. A pipeline may extract data from a source, transform "
                "it, validate its quality, and load it into a destination."
            ),
        ),
        (
            "Batch processing",
            (
                "Batch processing handles accumulated records in scheduled groups. "
                "It is useful for nightly reporting, historical analysis, and "
                "large transformations that do not require immediate results."
            ),
        ),
        (
            "Data quality",
            (
                "Data quality checks can detect missing values, duplicate records, "
                "invalid formats, and inconsistent identifiers before information "
                "is used by downstream applications."
            ),
        ),
    ],
    "rag_guide.pdf": [
        (
            "Retrieval augmented generation",
            (
                "Retrieval augmented generation, or RAG, retrieves relevant "
                "passages from a document collection and supplies them to a "
                "language model to help generate a grounded answer."
            ),
        ),
        (
            "Embeddings and semantic search",
            (
                "An embedding represents text as a numerical vector. Semantic "
                "search compares vector representations to find passages that "
                "are conceptually related, even when wording differs."
            ),
        ),
        (
            "Limitations of RAG",
            (
                "RAG quality depends on document coverage, chunk boundaries, "
                "retrieval ranking, and the language model's use of retrieved "
                "evidence. Retrieved context does not guarantee a correct answer."
            ),
        ),
    ],
}


def create_pdf(path: Path, pages: list[tuple[str, str]]) -> None:
    """Create a readable PDF with one topic per page."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with pymupdf.open() as document:
        for title, content in pages:
            page = document.new_page()
            page.insert_text((72, 72), title, fontsize=16)

            remaining_space = page.insert_textbox(
                pymupdf.Rect(72, 110, 540, 740),
                content,
                fontsize=12,
                lineheight=1.5,
            )

            if remaining_space < 0:
                raise ValueError(f"Text does not fit on page: {title}")

        document.save(path)


def main() -> None:
    """Generate all demonstration documents."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, pages in DOCUMENTS.items():
        path = OUTPUT_DIR / filename
        create_pdf(path, pages)
        print(f"Created {path} ({len(pages)} pages)")

    print(f"\nCorpus created in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
