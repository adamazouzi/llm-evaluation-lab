
"""FastAPI application for the LLM Evaluation Lab."""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from backend.app.schemas.rag import AskRequest, AskResponse
from backend.app.services.azure_openai import AzureOpenAIAnswerGenerator
from backend.app.services.document_loader import load_pdf_chunks
from backend.app.services.keyword_search import TfidfSearch
from backend.app.services.rag_pipeline import RAGPipeline
from backend.app.services.vector_search import VectorSearch

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_CORPUS_DIR = PROJECT_ROOT / "data" / "demo" / "multi_doc_corpus"


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


def create_pdf_retriever(retriever_type: str):
    """Load the PDF corpus and build the selected retrieval index."""
    if not PDF_CORPUS_DIR.exists():
        raise FileNotFoundError(
            f"PDF corpus directory not found: {PDF_CORPUS_DIR}"
        )

    pdf_files = sorted(PDF_CORPUS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in: {PDF_CORPUS_DIR}"
        )

    chunks = []
    metadata = []

    for pdf_path in pdf_files:
        document_chunks = load_pdf_chunks(pdf_path)

        for chunk in document_chunks:
            chunks.append(chunk.text)
            metadata.append(
                {
                    "chunk_id": (
                        f"{pdf_path.stem}-page-{chunk.page_number}"
                        f"-chunk-{chunk.chunk_index}"
                    ),
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                }
            )

    if retriever_type == "tfidf":
        retriever = TfidfSearch()
    elif retriever_type == "semantic":
        retriever = VectorSearch()
    else:
        raise ValueError(
            "Unsupported RETRIEVER_TYPE. Choose 'tfidf' or 'semantic'."
        )

    retriever.index(chunks, metadata)

    return retriever


def create_rag_pipeline() -> RAGPipeline:
    """Build the RAG pipeline using the configured corpus and provider."""
    provider = os.getenv("LLM_PROVIDER", "demo").strip().lower()
    corpus = os.getenv("RAG_CORPUS", "demo").strip().lower()
    retriever_type = (
        os.getenv("RETRIEVER_TYPE", "tfidf").strip().lower()
    )

    if provider not in {"demo", "azure"}:
        raise ValueError(
            "Unsupported LLM_PROVIDER. Choose 'demo' or 'azure'."
        )

    if corpus not in {"demo", "pdf"}:
        raise ValueError(
            "Unsupported RAG_CORPUS. Choose 'demo' or 'pdf'."
        )

    if corpus == "demo":
        pipeline = create_demo_pipeline()
    else:
        retriever = create_pdf_retriever(retriever_type)
        pipeline = RAGPipeline(
            retriever=retriever,
            answer_generator=demo_answer_generator,
            top_k=3,
        )

    if provider == "azure":
        pipeline.answer_generator = (
            AzureOpenAIAnswerGenerator.from_environment()
        )

    return pipeline


app = FastAPI(
    title="LLM Evaluation Lab API",
    description=(
        "API prototype for retrieval and RAG evaluation. "
        "Supports demo and PDF corpora, TF-IDF and semantic retrieval, "
        "and optional Azure OpenAI generation."
    ),
    version="0.4.0",
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
