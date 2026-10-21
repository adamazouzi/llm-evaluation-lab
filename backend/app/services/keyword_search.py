"""Lexical retrieval using TF-IDF."""

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class SearchResult:
    """A retrieved text chunk with its similarity score and metadata."""

    text: str
    score: float
    metadata: dict


class TfidfSearch:
    """Index text chunks and retrieve them using lexical similarity."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer()
        self.chunks: list[str] = []
        self.metadata: list[dict] = []
        self.matrix = None

    def index(
        self,
        chunks: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Build a TF-IDF index and associate stable IDs with chunks."""
        if not chunks:
            raise ValueError("chunks must not be empty")

        if any(
            not isinstance(chunk, str) or not chunk.strip()
            for chunk in chunks
        ):
            raise ValueError("chunks must contain non-empty strings")

        if metadata is not None and len(metadata) != len(chunks):
            raise ValueError("metadata and chunks must have the same length")

        self.chunks = list(chunks)

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

        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Return the top-k chunks ranked by TF-IDF cosine similarity."""
        if self.matrix is None:
            raise RuntimeError("Index some chunks before searching")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()

        indices = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]

        return [
            SearchResult(
                text=self.chunks[index],
                score=float(np.clip(scores[index], 0.0, 1.0)),
                metadata=self.metadata[index],
            )
            for index in indices
        ]
