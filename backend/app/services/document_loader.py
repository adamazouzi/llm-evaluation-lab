from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class DocumentChunk:
    """A text chunk with document and page metadata."""

    document_name: str
    page_number: int
    chunk_index: int
    text: str


def extract_pdf_text(pdf_path: str | Path) -> list[dict[str, str | int]]:
    """Extract non-empty text from a PDF, preserving page numbers."""
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The input file must have a .pdf extension.")

    pages: list[dict[str, str | int]] = []

    try:
        with pymupdf.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text().strip()
                if text:
                    pages.append(
                        {
                            "page_number": page_number,
                            "text": text,
                        }
                    )
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {path.name}") from exc

    if not pages:
        raise ValueError("The PDF contains no extractable text.")

    return pages


def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[str]:
    """Split text into character-based chunks with configurable overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    text = " ".join(text.split())
    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


def load_pdf_chunks(
    pdf_path: str | Path,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[DocumentChunk]:
    """Extract a PDF and return chunks with page-level metadata."""
    path = Path(pdf_path)
    pages = extract_pdf_text(path)
    chunks: list[DocumentChunk] = []
    global_index = 0

    for page in pages:
        page_chunks = chunk_text(
            str(page["text"]),
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in page_chunks:
            chunks.append(
                DocumentChunk(
                    document_name=path.name,
                    page_number=int(page["page_number"]),
                    chunk_index=global_index,
                    text=chunk,
                )
            )
            global_index += 1

    return chunks
