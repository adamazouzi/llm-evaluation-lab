from fastapi import FastAPI

app = FastAPI(
    title="LLM Evaluation Lab API",
    description="API for evaluating and comparing RAG systems.",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Return the application health status."""
    return {"status": "ok"}
