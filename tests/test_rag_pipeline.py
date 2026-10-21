
"""Unit tests for the RAG pipeline, without external API calls."""

from dataclasses import dataclass

import pytest

from backend.app.services.rag_pipeline import RAGPipeline


@dataclass
class FakeSearchResult:
    text: str
    score: float
    metadata: dict


class FakeRetriever:
    def __init__(self, results: list[FakeSearchResult]) -> None:
        self.results = results
        self.last_query = None
        self.last_top_k = None

    def search(self, query: str, top_k: int = 3) -> list[FakeSearchResult]:
        self.last_query = query
        self.last_top_k = top_k
        return self.results[:top_k]


def make_result(
    text: str,
    *,
    chunk_id: str = "chunk-1",
    score: float = 0.9,
    document_name: str = "guide.pdf",
    page_number: int = 2,
) -> FakeSearchResult:
    return FakeSearchResult(
        text=text,
        score=score,
        metadata={
            "chunk_id": chunk_id,
            "document_name": document_name,
            "page_number": page_number,
        },
    )


def test_pipeline_generates_answer_from_retrieved_context() -> None:
    retriever = FakeRetriever(
        [make_result("Python is a programming language.")]
    )
    captured = {}

    def generator(question: str, context: str) -> str:
        captured["question"] = question
        captured["context"] = context
        return "Python is a programming language."

    pipeline = RAGPipeline(retriever, generator, top_k=2)
    response = pipeline.answer("What is Python?")

    assert response.answer == "Python is a programming language."
    assert response.abstained is False
    assert len(response.sources) == 1
    assert response.sources[0].chunk_id == "chunk-1"
    assert captured["question"] == "What is Python?"
    assert "Python is a programming language." in captured["context"]
    assert "guide.pdf" in captured["context"]
    assert "page 2" in captured["context"]
    assert retriever.last_query == "What is Python?"
    assert retriever.last_top_k == 2
    assert response.latency_ms >= 0


def test_pipeline_abstains_when_no_results_are_retrieved() -> None:
    retriever = FakeRetriever([])
    generator_called = False

    def generator(question: str, context: str) -> str:
        nonlocal generator_called
        generator_called = True
        return "This should not be called."

    pipeline = RAGPipeline(retriever, generator)
    response = pipeline.answer("What is Python?")

    assert response.abstained is True
    assert response.sources == []
    assert "don't have enough relevant information" in response.answer
    assert generator_called is False


def test_pipeline_filters_results_below_minimum_score() -> None:
    retriever = FakeRetriever(
        [
            make_result("Low relevance passage.", chunk_id="low", score=0.1),
            make_result("Relevant passage.", chunk_id="high", score=0.8),
        ]
    )
    pipeline = RAGPipeline(
        retriever,
        lambda question, context: "Answer from context.",
        top_k=2,
        min_score=0.5,
    )

    response = pipeline.answer("Question")

    assert [source.chunk_id for source in response.sources] == ["high"]
    assert response.abstained is False


def test_pipeline_rejects_empty_question() -> None:
    pipeline = RAGPipeline(
        FakeRetriever([]),
        lambda question, context: "Answer",
    )

    with pytest.raises(ValueError, match="question must not be empty"):
        pipeline.answer("   ")


def test_pipeline_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        RAGPipeline(
            FakeRetriever([]),
            lambda question, context: "Answer",
            top_k=0,
        )


def test_pipeline_rejects_negative_minimum_score() -> None:
    with pytest.raises(ValueError, match="min_score must be non-negative"):
        RAGPipeline(
            FakeRetriever([]),
            lambda question, context: "Answer",
            min_score=-0.1,
        )


def test_pipeline_rejects_empty_generated_answer() -> None:
    pipeline = RAGPipeline(
        FakeRetriever([make_result("Relevant context.")]),
        lambda question, context: "   ",
    )

    with pytest.raises(
        ValueError,
        match="answer_generator must return a non-empty string",
    ):
        pipeline.answer("Question")
