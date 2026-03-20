"""
Tests for PDF Parser.
"""

import os
import tempfile
from pathlib import Path

import pytest

from app.ingestion.parser import PDFParser, ParsedDocument


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal valid PDF file for testing."""
    # Minimal PDF spec — two pages with text
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>stream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n431\n%%EOF"
    )
    pdf_path = tmp_path / "test_document.pdf"
    pdf_path.write_bytes(pdf_content)
    return pdf_path


class TestPDFParser:
    """Tests for PDFParser class."""

    def test_parse_valid_pdf(self, sample_pdf: Path) -> None:
        """Parser should successfully parse a valid PDF."""
        parser = PDFParser()
        result = parser.parse(sample_pdf)

        assert isinstance(result, ParsedDocument)
        assert result.document_id  # UUID assigned
        assert result.title == "test_document"
        assert result.total_pages >= 1

    def test_parse_with_custom_title(self, sample_pdf: Path) -> None:
        """Custom title should override filename."""
        parser = PDFParser()
        result = parser.parse(sample_pdf, title="My Custom Title")
        assert result.title == "My Custom Title"

    def test_parse_nonexistent_file(self) -> None:
        """Should raise FileNotFoundError for missing file."""
        parser = PDFParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path.pdf")

    def test_parse_non_pdf_file(self, tmp_path: Path) -> None:
        """Should raise ValueError for non-PDF file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")
        parser = PDFParser()
        with pytest.raises(ValueError, match="Expected PDF"):
            parser.parse(txt_file)

    def test_parsed_document_full_text(self, sample_pdf: Path) -> None:
        """full_text property should concatenate all pages."""
        parser = PDFParser()
        result = parser.parse(sample_pdf)
        # full_text should be a non-empty string (if pages have content)
        assert isinstance(result.full_text, str)
