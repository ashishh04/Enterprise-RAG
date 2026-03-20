"""
Tests for FAISS Vector Store.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.vectorstore.faiss_store import FAISSVectorStore, SearchResult


def _random_vectors(n: int, dim: int = 384) -> list[np.ndarray]:
    """Generate n random L2-normalized vectors."""
    vecs = []
    for _ in range(n):
        v = np.random.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        vecs.append(v)
    return vecs


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore class."""

    def test_add_and_search(self) -> None:
        """Should add vectors and find similar ones."""
        store = FAISSVectorStore(dimension=384)

        vecs = _random_vectors(10)
        metadata = [
            {"chunk_id": f"chunk-{i}", "text": f"text-{i}", "title": "doc"}
            for i in range(10)
        ]

        store.add_documents(vecs, metadata)
        assert store.size == 10

        results = store.search(vecs[0], top_k=3)
        assert len(results) == 3
        assert isinstance(results[0], SearchResult)
        # The most similar should be itself
        assert results[0].chunk_id == "chunk-0"
        assert results[0].score > 0.99

    def test_empty_search(self) -> None:
        """Search on empty index should return empty list."""
        store = FAISSVectorStore(dimension=384)
        query = _random_vectors(1)[0]
        results = store.search(query, top_k=5)
        assert results == []

    def test_score_threshold(self) -> None:
        """Threshold should filter low-score results."""
        store = FAISSVectorStore(dimension=384)
        vecs = _random_vectors(5)
        metadata = [
            {"chunk_id": f"c-{i}", "text": f"t-{i}"} for i in range(5)
        ]
        store.add_documents(vecs, metadata)

        # With very high threshold, should get fewer results
        results = store.search(vecs[0], top_k=5, score_threshold=0.99)
        assert len(results) <= 2  # Only nearly identical vectors

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Should persist and reload index correctly."""
        store1 = FAISSVectorStore(dimension=128, index_dir=str(tmp_path))
        vecs = _random_vectors(20, dim=128)
        metadata = [{"chunk_id": f"c-{i}", "text": f"t-{i}"} for i in range(20)]

        store1.add_documents(vecs, metadata)
        store1.save()

        # Load in a new instance
        store2 = FAISSVectorStore(dimension=128, index_dir=str(tmp_path))
        loaded = store2.load()
        assert loaded is True
        assert store2.size == 20

        # Search should work on loaded index
        results = store2.search(vecs[5], top_k=1)
        assert results[0].chunk_id == "c-5"

    def test_mismatched_lengths(self) -> None:
        """Should raise ValueError for mismatched vectors/metadata."""
        store = FAISSVectorStore(dimension=384)
        vecs = _random_vectors(3)
        metadata = [{"chunk_id": "c-0", "text": "t-0"}]  # Only 1

        with pytest.raises(ValueError, match="same length"):
            store.add_documents(vecs, metadata)

    def test_clear(self) -> None:
        """Clear should reset the index."""
        store = FAISSVectorStore(dimension=384)
        vecs = _random_vectors(5)
        metadata = [{"chunk_id": f"c-{i}", "text": f"t-{i}"} for i in range(5)]
        store.add_documents(vecs, metadata)
        assert store.size == 5

        store.clear()
        assert store.size == 0

    def test_metadata_filtering(self) -> None:
        """Should filter results by metadata key-value pairs."""
        store = FAISSVectorStore(dimension=384)
        vecs = _random_vectors(6)
        metadata = [
            {"chunk_id": f"c-{i}", "text": f"t-{i}", "category": "A" if i < 3 else "B"}
            for i in range(6)
        ]

        store.add_documents(vecs, metadata)

        results = store.search(
            vecs[0], top_k=6, metadata_filter={"category": "A"}
        )
        for r in results:
            assert r.metadata.get("category") == "A"
