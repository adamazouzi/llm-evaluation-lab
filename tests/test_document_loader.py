import pymupdf
import pytest

from backend.app.services.document_loader import (
    chunk_text,
    extract_pdf_text,
    load_pdf_chunks,
)


def create_test_pdf(path, page_texts):
    with pymupdf.open() as document:
        for text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), text)
        document.save(path)


def test_extract_pdf_text_preserves_page_numbers(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_test_pdf(pdf_path, ["First page content", "Second page content"])

    pages = extract_pdf_text(pdf_path)

    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2
    assert pages[0]["text"] == "First page content"


def test_chunk_text_respects_size_and_overlap():
    chunks = chunk_text("abcdefghij", chunk_size=6, overlap=2)

    assert chunks == ["abcdef", "efghij"]


def test_chunk_text_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=0)

    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=5, overlap=5)


def test_load_pdf_chunks_includes_metadata(tmp_path):
    pdf_path = tmp_path / "course.pdf"
    create_test_pdf(pdf_path, ["A short course passage"])

    chunks = load_pdf_chunks(pdf_path, chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].document_name == "course.pdf"
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "A short course passage"


def test_extract_pdf_text_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_pdf_text(tmp_path / "missing.pdf")


def test_extract_pdf_text_rejects_empty_pdf(tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    create_test_pdf(pdf_path, [""])

    with pytest.raises(ValueError, match="no extractable text"):
        extract_pdf_text(pdf_path)
