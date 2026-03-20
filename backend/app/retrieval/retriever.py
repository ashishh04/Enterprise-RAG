"""
Retrieval Pipeline — Query embedding, vector search, and context construction.

Orchestrates the embedding service and FAISS vector store to convert
a user query into a ranked set of relevant document chunks, then
constructs a token-bounded context window with citation metadata.
"""

import logging
import time
from dataclasses import dataclass, field

import tiktoken

from app.embeddings.embedder import EmbeddingService
from app.vectorstore.faiss_store import FAISSVectorStore, SearchResult

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    chunks: list[SearchResult]
    context_text: str
    citations: list[dict]
    latency_ms: float
    query: str = ""


class RetrievalPipeline:
    """
    End-to-end retrieval: query → embed → search → rerank → context.

    Features:
    - Embeds user query via EmbeddingService
    - Searches FAISS vector store
    - Reranks by similarity score
    - Constructs context window bounded by max_context_tokens
    - Preserves citation metadata for downstream generation
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
        top_k: int = 5,
        score_threshold: float = 0.3,
        max_context_tokens: int = 3000,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._max_context_tokens = max_context_tokens

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResult:
        """
        Execute the full retrieval pipeline for a query.

        Args:
            query: User question text.
            top_k: Override default top-k.
            score_threshold: Override default threshold.

        Returns:
            RetrievalResult with ranked chunks, context text, and citations.
        """
        start = time.perf_counter()
        k = top_k or self._top_k
        threshold = score_threshold or self._score_threshold

        # Step 1: Embed the query
        query_vector = self._embedding_service.embed_query(query)

        # Step 2: Vector search
        results = self._vector_store.search(
            query_vector=query_vector,
            top_k=k,
            score_threshold=threshold,
        )

        # Step 3: Rerank by score (already sorted by FAISS, but ensure determinism)
        results.sort(key=lambda r: r.score, reverse=True)

        # Step 4: Construct context window
        context_text, citations, included_chunks = self._build_context(results)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Retrieval completed: query='%s', results=%d, context_tokens=%d, latency=%.1fms",
            query[:60],
            len(included_chunks),
            self._count_tokens(context_text),
            elapsed_ms,
        )

        return RetrievalResult(
            chunks=included_chunks,
            context_text=context_text,
            citations=citations,
            latency_ms=elapsed_ms,
            query=query,
        )

    def _build_context(
        self, results: list[SearchResult]
    ) -> tuple[list[dict], list[SearchResult]]:
        """
        Build token-bounded context from search results.

        Returns:
            Tuple of (context_text, citations_list, included_chunks).
        """
        context_parts: list[str] = []
        citations: list[dict] = []
        included: list[SearchResult] = []
        total_tokens = 0

        for result in results:
            chunk_tokens = self._count_tokens(result.text)

            if total_tokens + chunk_tokens > self._max_context_tokens:
                logger.debug(
                    "Context token limit reached (%d/%d) — stopping at %d chunks.",
                    total_tokens,
                    self._max_context_tokens,
                    len(included),
                )
                break

            # Format context block with citation marker
            source_name = result.metadata.get("title", "Unknown")
            page_num = result.metadata.get("page", "?")
            citation_ref = f"[{source_name} – Page {page_num}]"

            context_parts.append(f"{citation_ref}\n{result.text}")
            citations.append(
                {
                    "document": source_name,
                    "page": page_num,
                    "chunk_id": result.chunk_id,
                    "score": round(result.score, 4),
                    "source": result.metadata.get("source", ""),
                }
            )
            included.append(result)
            total_tokens += chunk_tokens

        context_text = "\n\n---\n\n".join(context_parts)
        return context_text, citations, included

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Count tokens using cl100k_base encoding."""
        return len(_ENCODING.encode(text))
