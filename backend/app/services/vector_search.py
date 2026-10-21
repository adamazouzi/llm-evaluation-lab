
"""Local semantic search using sentence embeddings."""

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class SearchResult:
    """A retrieved text chunk with its similarity score and metadata."""

    text: str
    score: float
    metadata: dict


class VectorSearch:
    """Encode text chunks and retrieve the most similar ones."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.chunks: list[str] = []
        self.metadata: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def index(
        self,
        chunks: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Encode chunks and associate each one with a stable identifier."""
        if not chunks:
            raise ValueError("chunks must not be empty")

        if any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks):
            raise ValueError("chunks must contain non-empty strings")

        if metadata is not None and len(metadata) != len(chunks):
            raise ValueError("metadata and chunks must have the same length")

        self.chunks = chunks

        if metadata is None:
            self.metadata = [
                {"chunk_id": f"chunk-{i}"}
                for i in range(len(chunks))
            ]
        else:
            self.metadata = []

            for i, item in enumerate(metadata):
                chunk_metadata = dict(item)

                if not chunk_metadata.get("chunk_id"):
                    chunk_metadata["chunk_id"] = f"chunk-{i}"

                self.metadata.append(chunk_metadata)

        self.embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Return the top-k chunks ranked by cosine similarity."""
        if self.embeddings is None:
            raise RuntimeError("Index some chunks before searching")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = self.embeddings @ query_embedding
        indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(
                text=self.chunks[int(i)],
                score=float(scores[i]),
                metadata=self.metadata[int(i)],
            )
            for i in indices
        ]
