"""
FAISS Vector Store — In-memory vector index with persistent storage.

Maintains a FAISS IndexFlatIP index (inner product on L2-normalized
vectors = cosine similarity) with a parallel metadata store.
Thread-safe for concurrent reads/writes.
"""

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with score and metadata."""

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class FAISSVectorStore:
    """
    FAISS-backed vector store with metadata persistence.

    Features:
    - IndexFlatIP for exact inner-product search (cosine with normalized vecs)
    - Parallel metadata list stored as JSON
    - Thread-safe via threading.Lock
    - Persistent save/load to disk
    - Score threshold and metadata filtering
    """

    def __init__(self, dimension: int, index_dir: Optional[str | Path] = None) -> None:
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._index_dir: Optional[Path] = Path(index_dir) if index_dir else None

        if self._index_dir:
            self._index_path = self._index_dir / "index.faiss"
            self._metadata_path = self._index_dir / "metadata.json"
        else:
            self._index_path = None
            self._metadata_path = None

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal

    @property
    def dimension(self) -> int:
        """Vector dimension."""
        return self._dimension

    def add_documents(
        self,
        vectors: list[np.ndarray],
        metadata_list: list[dict[str, Any]],
    ) -> None:
        """
        Add vectors and their metadata to the index.

        Args:
            vectors: List of L2-normalized embedding vectors.
            metadata_list: Parallel list of metadata dicts (must include 'chunk_id' and 'text').
        """
        if len(vectors) != len(metadata_list):
            raise ValueError(
                f"vectors ({len(vectors)}) and metadata ({len(metadata_list)}) must have same length"
            )

        if not vectors:
            return

        matrix = np.vstack(vectors).astype(np.float32)

        with self._lock:
            self._index.add(matrix)
            self._metadata.extend(metadata_list)

        logger.info("Added %d vectors. Total index size: %d", len(vectors), self.size)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Search the index for nearest neighbors.

        Args:
            query_vector: L2-normalized query embedding.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score.
            metadata_filter: Optional key-value pairs to filter results.

        Returns:
            List of SearchResult objects sorted by descending score.
        """
        if self.size == 0:
            logger.warning("Search on empty index — returning empty results.")
            return []

        query = query_vector.reshape(1, -1).astype(np.float32)

        # Search more than top_k to allow for filtering
        search_k = min(top_k * 3, self.size) if metadata_filter else min(top_k, self.size)

        with self._lock:
            scores, indices = self._index.search(query, search_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            if score < score_threshold:
                continue

            meta = self._metadata[idx]

            # Apply metadata filter
            if metadata_filter:
                if not all(meta.get(k) == v for k, v in metadata_filter.items()):
                    continue

            results.append(
                SearchResult(
                    chunk_id=meta.get("chunk_id", ""),
                    score=float(score),
                    text=meta.get("text", ""),
                    metadata=meta,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if not self._index_dir:
            logger.warning("No index directory configured — skipping save.")
            return

        self._index_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            faiss.write_index(self._index, str(self._index_path))
            with open(self._metadata_path, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f)

        logger.info("Saved FAISS index (%d vectors) to %s", self.size, self._index_dir)

    def load(self) -> bool:
        """
        Load index and metadata from disk.

        Returns:
            True if loaded successfully, False if files not found.
        """
        if not self._index_path or not self._metadata_path:
            return False

        if not self._index_path.exists() or not self._metadata_path.exists():
            logger.info("No existing index found at %s", self._index_dir)
            return False

        try:
            with self._lock:
                self._index = faiss.read_index(str(self._index_path))
                with open(self._metadata_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)

            logger.info(
                "Loaded FAISS index: %d vectors from %s",
                self.size,
                self._index_dir,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load index: %s", exc)
            return False

    def clear(self) -> None:
        """Reset the index to empty."""
        with self._lock:
            self._index = faiss.IndexFlatIP(self._dimension)
            self._metadata = []
        logger.info("FAISS index cleared.")
