
"""Request and response schemas for the RAG API."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Question submitted to the RAG endpoint."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question to answer using the indexed documents.",
    )


class SourceResponse(BaseModel):
    """A source passage used to answer a question."""

    chunk_id: str
    text: str
    score: float
    document_name: str | None = None
    page_number: int | None = None


class AskResponse(BaseModel):
    """Answer and retrieval metadata returned by the API."""

    question: str
    answer: str
    sources: list[SourceResponse]
    latency_ms: float
    abstained: bool
