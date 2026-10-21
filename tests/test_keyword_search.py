"""Tests for the TF-IDF lexical retriever."""

import pytest

from backend.app.services.keyword_search import TfidfSearch


def test_search_ranks_matching_chunk_first() -> None:
    """A lexical match should rank above a document with no matching terms."""
    retriever = TfidfSearch()
    retriever.index(
        [
            "Python is used for data analysis",
            "Deep learning uses neural networks",
        ]
    )

    results = retriever.search("neural networks", top_k=2)

    assert len(results) == 2
    assert results[0].metadata["chunk_id"] == "chunk-1"
    assert results[0].score > results[1].score
    assert results[1].score == 0.0


def test_index_preserves_supplied_metadata() -> None:
    """Supplied chunk IDs and metadata should be preserved."""
    retriever = TfidfSearch()
    metadata = [
        {"chunk_id": "doc-a", "page_number": 1},
        {"chunk_id": "doc-b", "page_number": 2},
    ]
    retriever.index(
        ["Python programming language", "Neural network architecture"],
        metadata,
    )

    results = retriever.search("Python programming", top_k=1)

    assert results[0].metadata == {
        "chunk_id": "doc-a",
        "page_number": 1,
    }


def test_index_rejects_empty_chunks() -> None:
    """An empty collection cannot be indexed."""
    retriever = TfidfSearch()

    with pytest.raises(ValueError, match="must not be empty"):
        retriever.index([])


def test_index_rejects_blank_chunk() -> None:
    """Every chunk must contain non-whitespace text."""
    retriever = TfidfSearch()

    with pytest.raises(ValueError, match="non-empty strings"):
        retriever.index(["valid text", "   "])


def test_index_rejects_mismatched_metadata() -> None:
    """Metadata and chunks must have equal lengths."""
    retriever = TfidfSearch()

    with pytest.raises(ValueError, match="same length"):
        retriever.index(["one document"], [])


def test_search_requires_index() -> None:
    """Searching before indexing should raise a clear error."""
    retriever = TfidfSearch()

    with pytest.raises(RuntimeError, match="Index some chunks"):
        retriever.search("some query")


def test_search_rejects_blank_query() -> None:
    """A query must contain non-whitespace text."""
    retriever = TfidfSearch()
    retriever.index(["Python programming language"])

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.search("   ")


def test_search_rejects_invalid_top_k() -> None:
    """The requested number of results must be positive."""
    retriever = TfidfSearch()
    retriever.index(["Python programming language"])

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.search("Python", top_k=0)
