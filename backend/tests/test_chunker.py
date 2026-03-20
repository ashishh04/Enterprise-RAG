"""
Tests for Semantic Chunker.
"""

import pytest

from app.ingestion.chunker import SemanticChunker, Chunk


class TestSemanticChunker:
    """Tests for SemanticChunker class."""

    def test_basic_chunking(self) -> None:
        """Should create chunks from text."""
        chunker = SemanticChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=5)

        # Create a text with multiple paragraphs
        paragraphs = ["This is paragraph number {i}. " * 5 for i in range(10)]
        text = "\n\n".join(paragraphs)

        # Create a mock document
        from app.ingestion.parser import ParsedDocument, PageContent

        doc = ParsedDocument(
            document_id="test-doc-id",
            title="Test Doc",
            source_path="/test/path.pdf",
            pages=[
                PageContent(
                    page_number=1,
                    text=text,
                    metadata={"title": "Test Doc", "page": 1},
                )
            ],
            total_pages=1,
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.document_id == "test-doc-id"
            assert chunk.chunk_id  # UUID assigned
            assert chunk.text.strip()  # non-empty
            assert chunk.token_count > 0

    def test_overlap_validation(self) -> None:
        """overlap must be less than chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            SemanticChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_metadata_preserved(self) -> None:
        """Each chunk should carry source metadata."""
        chunker = SemanticChunker(chunk_size=200, chunk_overlap=30, min_chunk_size=5)

        from app.ingestion.parser import ParsedDocument, PageContent

        doc = ParsedDocument(
            document_id="doc-123",
            title="Report",
            source_path="/docs/report.pdf",
            pages=[
                PageContent(
                    page_number=3,
                    text="Some text content with enough words to make a chunk. " * 10,
                    metadata={"title": "Report", "page": 3, "source": "/docs/report.pdf"},
                )
            ],
            total_pages=5,
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata.get("title") == "Report"
            assert chunk.metadata.get("page") == 3

    def test_empty_document(self) -> None:
        """Empty document should produce no chunks."""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)

        from app.ingestion.parser import ParsedDocument

        doc = ParsedDocument(
            document_id="empty-doc",
            title="Empty",
            source_path="/empty.pdf",
            pages=[],
            total_pages=0,
        )

        chunks = chunker.chunk_document(doc)
        assert chunks == []
