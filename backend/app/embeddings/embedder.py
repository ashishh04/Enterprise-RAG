"""
Embedding Service — Local embeddings via sentence-transformers.

Generates vector embeddings using a locally loaded sentence-transformers
model. Includes content-hash caching to prevent recomputation and
L2 normalization for cosine similarity compatibility with FAISS IndexFlatIP.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates embeddings using sentence-transformers, running locally.

    Features:
    - Local inference (no API calls, no API key required)
    - Content-hash caching on disk
    - L2 normalization for cosine similarity
    - Batch encoding
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str | Path] = None,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._cache_path: Optional[Path] = None
        self._cache: dict[str, list[float]] = {}

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Model loaded — dimension=%d, device=%s",
            self._dimension,
            self._model.device,
        )

        if cache_dir:
            self._cache_path = Path(cache_dir) / "embedding_cache.json"
            self._load_cache()

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """
        Generate embeddings for a list of texts.

        Uses cache for previously seen content. New texts are batch-encoded
        and then cached.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of L2-normalized numpy vectors.
        """
        results: list[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache
        for i, text in enumerate(texts):
            content_hash = self._hash_text(text)
            if content_hash in self._cache:
                results[i] = np.array(self._cache[content_hash], dtype=np.float32)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        cache_hits = len(texts) - len(uncached_texts)
        if cache_hits > 0:
            logger.debug("Embedding cache: %d hits, %d misses.", cache_hits, len(uncached_texts))

        # Encode uncached texts
        if uncached_texts:
            logger.info("Encoding %d texts (batch_size=%d)...", len(uncached_texts), self._batch_size)
            raw_embeddings = self._model.encode(
                uncached_texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,  # L2 normalize
                convert_to_numpy=True,
            )

            for j, idx in enumerate(uncached_indices):
                vec = raw_embeddings[j].astype(np.float32)
                results[idx] = vec
                # Cache the result
                content_hash = self._hash_text(uncached_texts[j])
                self._cache[content_hash] = vec.tolist()

            self._save_cache()

        return [r for r in results if r is not None]

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Args:
            query: The query text.

        Returns:
            L2-normalized numpy vector.
        """
        result = self.embed_texts([query])
        return result[0]

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_text(text: str) -> str:
        """SHA-256 content hash of text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        if self._cache_path and self._cache_path.exists():
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info("Loaded %d cached embeddings.", len(self._cache))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load embedding cache: %s", exc)
                self._cache = {}
        else:
            self._cache = {}

    def _save_cache(self) -> None:
        """Persist embedding cache to disk."""
        if self._cache_path:
            try:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._cache_path, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f)
                logger.debug("Saved %d embeddings to cache.", len(self._cache))
            except OSError as exc:
                logger.warning("Failed to save embedding cache: %s", exc)

    def clear_cache(self) -> None:
        """Clear the in-memory and on-disk cache."""
        self._cache = {}
        if self._cache_path and self._cache_path.exists():
            self._cache_path.unlink()
            logger.info("Embedding cache cleared.")
