"""
PDF Parser — Extracts clean text from PDF documents.

Uses pypdf to read pages and applies heuristic cleaning to remove
headers, footers, and normalize whitespace. Handles encoding errors
and empty pages gracefully.
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Extracted content from a single PDF page."""

    page_number: int
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Complete parsed PDF document."""

    document_id: str
    title: str
    source_path: str
    pages: list[PageContent]
    total_pages: int

    @property
    def full_text(self) -> str:
        """Concatenate all page texts."""
        return "\n\n".join(page.text for page in self.pages if page.text.strip())


class PDFParser:
    """
    Parses PDF files and extracts clean text content.

    Handles:
    - Per-page text extraction
    - Header/footer heuristic removal
    - Whitespace normalization
    - Encoding error recovery
    """

    def __init__(
        self,
        strip_headers_footers: bool = True,
        header_footer_lines: int = 2,
    ) -> None:
        self._strip_headers_footers = strip_headers_footers
        self._header_footer_lines = header_footer_lines

    def parse(self, file_path: str | Path, title: Optional[str] = None) -> ParsedDocument:
        """
        Parse a PDF file and return structured document content.

        Args:
            file_path: Path to the PDF file.
            title: Optional document title (defaults to filename).

        Returns:
            ParsedDocument with extracted page content.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file is not a valid PDF.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected PDF file, got: {file_path.suffix}")

        document_id = str(uuid.uuid4())
        doc_title = title or file_path.stem

        logger.info("Parsing PDF: %s (doc_id=%s)", file_path.name, document_id)

        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:
            logger.error("Failed to read PDF '%s': %s", file_path.name, exc)
            raise ValueError(f"Invalid or corrupted PDF: {file_path.name}") from exc

        pages: list[PageContent] = []
        total_pages = len(reader.pages)

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:
                logger.warning(
                    "Failed to extract text from page %d of '%s': %s",
                    page_num,
                    file_path.name,
                    exc,
                )
                raw_text = ""

            cleaned = self._clean_text(raw_text, page_num, total_pages)

            if cleaned.strip():
                pages.append(
                    PageContent(
                        page_number=page_num,
                        text=cleaned,
                        metadata={
                            "title": doc_title,
                            "source": str(file_path),
                            "page": page_num,
                            "total_pages": total_pages,
                        },
                    )
                )
            else:
                logger.debug(
                    "Page %d of '%s' is empty after cleaning — skipped.",
                    page_num,
                    file_path.name,
                )

        logger.info(
            "Parsed '%s': %d/%d pages with content.",
            file_path.name,
            len(pages),
            total_pages,
        )

        return ParsedDocument(
            document_id=document_id,
            title=doc_title,
            source_path=str(file_path),
            pages=pages,
            total_pages=total_pages,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clean_text(self, text: str, page_num: int, total_pages: int) -> str:
        """Clean extracted text: strip headers/footers, normalize whitespace."""
        if not text:
            return ""

        lines = text.split("\n")

        if self._strip_headers_footers and len(lines) > (self._header_footer_lines * 2 + 1):
            lines = self._remove_headers_footers(lines, page_num, total_pages)

        text = "\n".join(lines)

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)  # collapse horizontal whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)  # max 2 consecutive newlines
        text = text.strip()

        return text

    def _remove_headers_footers(
        self, lines: list[str], page_num: int, total_pages: int
    ) -> list[str]:
        """Heuristic removal of header/footer lines."""
        n = self._header_footer_lines

        # Check top lines for header patterns
        header_end = 0
        for i in range(min(n, len(lines))):
            line = lines[i].strip()
            if self._is_header_footer_line(line, page_num, total_pages):
                header_end = i + 1

        # Check bottom lines for footer patterns
        footer_start = len(lines)
        for i in range(len(lines) - 1, max(len(lines) - n - 1, header_end) - 1, -1):
            line = lines[i].strip()
            if self._is_header_footer_line(line, page_num, total_pages):
                footer_start = i

        return lines[header_end:footer_start]

    @staticmethod
    def _is_header_footer_line(line: str, page_num: int, total_pages: int) -> bool:
        """Check if a line looks like a header or footer."""
        if not line:
            return True

        lower = line.lower().strip()

        # Page number patterns
        if lower in (str(page_num), f"page {page_num}", f"- {page_num} -"):
            return True
        if re.match(r"^page\s+\d+\s*(of\s+\d+)?$", lower):
            return True
        if re.match(r"^\d+\s*/\s*\d+$", lower):
            return True

        # Very short lines at boundaries (likely headers)
        if len(lower) < 5 and lower.isdigit():
            return True

        return False
