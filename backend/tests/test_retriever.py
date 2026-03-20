"""
Tests for Retrieval Pipeline.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.retrieval.retriever import RetrievalPipeline, RetrievalResult
from app.vectorstore.faiss_store import SearchResult


def _mock_embedding_service():
    """Create a mock EmbeddingService."""
    mock = MagicMock()
    dim = 384
    vec = np.random.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    mock.embed_query.return_value = vec
    mock.dimension = dim
    return mock


def _mock_vector_store(results: list[SearchResult]):
    """Create a mock FAISSVectorStore returning given results."""
    mock = MagicMock()
    mock.search.return_value = results
    mock.size = len(results)
    return mock


class TestRetrievalPipeline:
    """Tests for RetrievalPipeline class."""

    def test_retrieve_returns_result(self) -> None:
        """Should return a RetrievalResult with chunks and context."""
        search_results = [
            SearchResult(
                chunk_id="c-1",
                score=0.92,
                text="Python is a programming language.",
                metadata={"title": "Python Guide", "page": 5, "source": "/docs/python.pdf"},
            ),
            SearchResult(
                chunk_id="c-2",
                score=0.85,
                text="Python supports OOP paradigms.",
                metadata={"title": "Python Guide", "page": 12, "source": "/docs/python.pdf"},
            ),
        ]

        emb = _mock_embedding_service()
        vs = _mock_vector_store(search_results)

        pipeline = RetrievalPipeline(
            embedding_service=emb,
            vector_store=vs,
            top_k=5,
            score_threshold=0.3,
            max_context_tokens=3000,
        )

        result = pipeline.retrieve("What is Python?")

        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) == 2
        assert len(result.citations) == 2
        assert result.context_text  # non-empty
        assert result.latency_ms > 0
        assert result.citations[0]["document"] == "Python Guide"
        assert result.citations[0]["page"] == 5

    def test_empty_results(self) -> None:
        """Should handle empty search results gracefully."""
        emb = _mock_embedding_service()
        vs = _mock_vector_store([])

        pipeline = RetrievalPipeline(
            embedding_service=emb,
            vector_store=vs,
        )

        result = pipeline.retrieve("nonexistent query")
        assert result.chunks == []
        assert result.context_text == ""
        assert result.citations == []

    def test_context_token_limit(self) -> None:
        """Should stop adding chunks when token limit is reached."""
        # Create results with known large text
        large_text = "word " * 2000  # ~2000 tokens
        search_results = [
            SearchResult(
                chunk_id=f"c-{i}",
                score=0.9 - i * 0.1,
                text=large_text,
                metadata={"title": "Big Doc", "page": i},
            )
            for i in range(5)
        ]

        emb = _mock_embedding_service()
        vs = _mock_vector_store(search_results)

        pipeline = RetrievalPipeline(
            embedding_service=emb,
            vector_store=vs,
            max_context_tokens=3000,
        )

        result = pipeline.retrieve("test")
        # Should include fewer than 5 chunks due to token limit
        assert len(result.chunks) < 5
