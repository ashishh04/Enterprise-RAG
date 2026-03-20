"""
Semantic Chunker — Splits document text into overlapping chunks.

Performs paragraph-aware splitting with token-accurate sizing using
tiktoken. Preserves semantic boundaries and generates metadata for
each chunk for downstream retrieval and citation.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

from app.ingestion.parser import PageContent, ParsedDocument

logger = logging.getLogger(__name__)

# Use the cl100k_base encoding (GPT-4 / text-embedding-3 tokenizer)
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A single text chunk with metadata for retrieval."""

    chunk_id: str
    document_id: str
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)


class SemanticChunker:
    """
    Splits document pages into semantically meaningful, overlapping chunks.

    Strategy:
    1. Split at paragraph boundaries (double newline).
    2. If a paragraph exceeds max size, split at sentence boundaries.
    3. Merge small consecutive paragraphs into one chunk.
    4. Apply token overlap between consecutive chunks.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        min_chunk_size: int = 50,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        """
        Split a parsed document into chunks.

        Args:
            document: A ParsedDocument from the PDF parser.

        Returns:
            List of Chunk objects with metadata.
        """
        all_chunks: list[Chunk] = []

        for page in document.pages:
            page_chunks = self._chunk_page(page, document.document_id)
            all_chunks.extend(page_chunks)

        logger.info(
            "Chunked document '%s': %d chunks from %d pages.",
            document.title,
            len(all_chunks),
            len(document.pages),
        )

        return all_chunks

    def _chunk_page(self, page: PageContent, document_id: str) -> list[Chunk]:
        """Chunk a single page into token-bounded segments."""
        paragraphs = self._split_paragraphs(page.text)

        # Split large paragraphs into sentences
        segments: list[str] = []
        for para in paragraphs:
            if self._count_tokens(para) > self._chunk_size:
                segments.extend(self._split_sentences(para))
            else:
                segments.append(para)

        # Merge segments into chunks respecting token limits
        return self._merge_into_chunks(segments, document_id, page.metadata)

    def _merge_into_chunks(
        self,
        segments: list[str],
        document_id: str,
        metadata: dict,
    ) -> list[Chunk]:
        """Merge text segments into overlapping chunks."""
        chunks: list[Chunk] = []
        current_segments: list[str] = []
        current_tokens = 0

        for segment in segments:
            seg_tokens = self._count_tokens(segment)

            # If single segment exceeds chunk_size, force split it
            if seg_tokens > self._chunk_size:
                # Flush current buffer first
                if current_segments:
                    chunks.append(
                        self._create_chunk(
                            "\n\n".join(current_segments),
                            document_id,
                            metadata,
                        )
                    )
                # Force-chunk the oversized segment by tokens
                chunks.extend(
                    self._force_split(segment, document_id, metadata)
                )
                current_segments = []
                current_tokens = 0
                continue

            if current_tokens + seg_tokens > self._chunk_size and current_segments:
                # Flush current chunk
                chunk_text = "\n\n".join(current_segments)
                chunks.append(self._create_chunk(chunk_text, document_id, metadata))

                # Build overlap: take trailing segments up to overlap token budget
                overlap_segments: list[str] = []
                overlap_tokens = 0
                for prev_seg in reversed(current_segments):
                    prev_tokens = self._count_tokens(prev_seg)
                    if overlap_tokens + prev_tokens > self._chunk_overlap:
                        break
                    overlap_segments.insert(0, prev_seg)
                    overlap_tokens += prev_tokens

                current_segments = overlap_segments
                current_tokens = overlap_tokens

            current_segments.append(segment)
            current_tokens += seg_tokens

        # Flush remaining
        if current_segments:
            text = "\n\n".join(current_segments)
            if self._count_tokens(text) >= self._min_chunk_size:
                chunks.append(self._create_chunk(text, document_id, metadata))

        return chunks

    def _force_split(
        self, text: str, document_id: str, metadata: dict
    ) -> list[Chunk]:
        """Force split a long text by token count."""
        tokens = _ENCODING.encode(text)
        chunks: list[Chunk] = []

        start = 0
        while start < len(tokens):
            end = min(start + self._chunk_size, len(tokens))
            chunk_text = _ENCODING.decode(tokens[start:end])
            if self._count_tokens(chunk_text) >= self._min_chunk_size:
                chunks.append(self._create_chunk(chunk_text, document_id, metadata))
            start = end - self._chunk_overlap if end < len(tokens) else end

        return chunks

    def _create_chunk(self, text: str, document_id: str, metadata: dict) -> Chunk:
        """Create a Chunk object with a unique ID."""
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            text=text.strip(),
            token_count=self._count_tokens(text),
            metadata={**metadata},
        )

    # ------------------------------------------------------------------
    # Text splitting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split text on double newlines, filtering empties."""
        parts = text.split("\n\n")
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentence-like segments."""
        import re

        # Split on sentence-ending punctuation followed by space or newline
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Count tokens using tiktoken cl100k_base encoding."""
        return len(_ENCODING.encode(text))
