import pytest

from backend.app.services.vector_search import VectorSearch


@pytest.fixture(scope="module")
def search_engine():
    engine = VectorSearch()
    engine.index(
        [
            "Python is a programming language.",
            "Paris is the capital of France.",
            "Machine learning uses data to learn patterns.",
        ],
        metadata=[
            {"source": "python.txt"},
            {"source": "geography.txt"},
            {"source": "ml.txt"},
        ],
    )
    return engine


def test_search_returns_relevant_result(search_engine):
    results = search_engine.search("What is machine learning?", top_k=1)

    assert len(results) == 1
    assert "Machine learning" in results[0].text
    assert results[0].metadata["source"] == "ml.txt"


def test_search_respects_top_k(search_engine):
    results = search_engine.search("programming and data", top_k=2)

    assert len(results) == 2


def test_search_rejects_empty_query(search_engine):
    with pytest.raises(ValueError):
        search_engine.search("   ")


def test_index_rejects_empty_chunks(search_engine):
    with pytest.raises(ValueError):
        search_engine.index([])


def test_search_requires_index():
    engine = VectorSearch()

    with pytest.raises(RuntimeError):
        engine.search("test")
