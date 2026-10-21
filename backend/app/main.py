
"""FastAPI application for the LLM Evaluation Lab."""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from backend.app.schemas.rag import AskRequest, AskResponse
from backend.app.services.azure_openai import AzureOpenAIAnswerGenerator
from backend.app.services.keyword_search import TfidfSearch
from backend.app.services.rag_pipeline import RAGPipeline

load_dotenv()


def demo_answer_generator(question: str, context: str) -> str:
    """Return retrieved context without pretending an LLM was used."""
    return (
        "Demo mode: no LLM has generated this answer. "
        "The following retrieved context is provided for inspection:\n\n"
        f"{context}"
    )


def create_demo_pipeline() -> RAGPipeline:
    """Build a small local index for offline demonstrations."""
    chunks = [
        (
            "Python is a programming language commonly used for "
            "automation, data analysis, and machine learning."
        ),
        (
            "Retrieval-Augmented Generation (RAG) retrieves relevant "
            "document passages and provides them as context to a language "
            "model when generating an answer."
        ),
        (
            "TF-IDF is a lexical retrieval method that represents text "
            "using term frequency and inverse document frequency."
        ),
    ]

    metadata = [
        {"chunk_id": "demo-python", "document_name": "demo-guide.txt"},
        {"chunk_id": "demo-rag", "document_name": "demo-guide.txt"},
        {"chunk_id": "demo-tfidf", "document_name": "demo-guide.txt"},
    ]

    retriever = TfidfSearch()
    retriever.index(chunks, metadata)

    return RAGPipeline(
        retriever=retriever,
        answer_generator=demo_answer_generator,
        top_k=3,
    )


def create_rag_pipeline() -> RAGPipeline:
    """Build the RAG pipeline using the selected provider."""
    provider = os.getenv("LLM_PROVIDER", "demo").strip().lower()

    if provider == "demo":
        return create_demo_pipeline()

    if provider == "azure":
        generator = AzureOpenAIAnswerGenerator.from_environment()
        pipeline = create_demo_pipeline()
        pipeline.answer_generator = generator
        return pipeline

    raise ValueError(
        "Unsupported LLM_PROVIDER. Choose 'demo' or 'azure'."
    )


app = FastAPI(
    title="LLM Evaluation Lab API",
    description=(
        "API prototype for retrieval and RAG evaluation. "
        "Azure OpenAI is optional; demo mode works offline."
    ),
    version="0.3.0",
)

app.state.rag_pipeline = create_rag_pipeline()


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Return the application health status."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask_question(request: AskRequest) -> AskResponse:
    """Retrieve passages and generate an answer using the selected provider."""
    try:
        result = app.state.rag_pipeline.answer(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The RAG pipeline failed to generate a response.",
        ) from exc

    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            {
                "chunk_id": source.chunk_id,
                "text": source.text,
                "score": source.score,
                "document_name": source.document_name,
                "page_number": source.page_number,
            }
            for source in result.sources
        ],
        latency_ms=result.latency_ms,
        abstained=result.abstained,
    )
