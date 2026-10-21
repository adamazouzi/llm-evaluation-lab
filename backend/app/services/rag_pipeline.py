
"""A small, testable Retrieval-Augmented Generation pipeline."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

AnswerGenerator = Callable[[str, str], str]

ABSTENTION_MESSAGE = (
    "I don't have enough relevant information in the "
    "available documents to answer this question."
)


@dataclass(frozen=True)
class Source:
    """A retrieved source passage with its document metadata."""

    chunk_id: str
    text: str
    score: float
    document_name: str | None = None
    page_number: int | None = None


@dataclass(frozen=True)
class RAGResponse:
    """The answer and traceable retrieval information."""

    question: str
    answer: str
    sources: list[Source]
    latency_ms: float
    abstained: bool


def build_context(sources: list[Source]) -> str:
    """Format retrieved passages for the answer generator."""
    sections = []

    for index, source in enumerate(sources, start=1):
        location = source.document_name or "Unknown document"
        if source.page_number is not None:
            location += f", page {source.page_number}"

        sections.append(
            f"[Source {index}: {location}; chunk_id={source.chunk_id}]\n"
            f"{source.text}"
        )

    return "\n\n".join(sections)


class RAGPipeline:
    """Retrieve relevant passages and generate an answer from their context."""

    def __init__(
        self,
        retriever: Any,
        answer_generator: AnswerGenerator,
        *,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if min_score < 0.0:
            raise ValueError("min_score must be non-negative")

        self.retriever = retriever
        self.answer_generator = answer_generator
        self.top_k = top_k
        self.min_score = min_score

    def answer(self, question: str) -> RAGResponse:
        """Answer a question using retrieved context and return its sources."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be empty")

        started_at = perf_counter()
        results = self.retriever.search(question, top_k=self.top_k)

        sources = [
            Source(
                chunk_id=str(
                    result.metadata.get("chunk_id", f"chunk-{index}")
                ),
                text=result.text,
                score=float(result.score),
                document_name=result.metadata.get("document_name"),
                page_number=result.metadata.get("page_number"),
            )
            for index, result in enumerate(results)
            if (
                isinstance(result.text, str)
                and result.text.strip()
                and float(result.score) >= self.min_score
            )
        ]

        if not sources:
            return RAGResponse(
                question=question,
                answer=ABSTENTION_MESSAGE,
                sources=[],
                latency_ms=(perf_counter() - started_at) * 1000,
                abstained=True,
            )

        context = build_context(sources)
        answer = self.answer_generator(question.strip(), context)

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer_generator must return a non-empty string")

        return RAGResponse(
            question=question,
            answer=answer.strip(),
            sources=sources,
            latency_ms=(perf_counter() - started_at) * 1000,
            abstained=False,
        )
