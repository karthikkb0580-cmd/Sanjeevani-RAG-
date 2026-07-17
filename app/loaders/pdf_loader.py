"""
Module 4: Document Loaders – PDF
app/loaders/pdf_loader.py

Extracts text from PDF files while preserving headings, paragraphs,
page numbers, and document-level metadata using PyMuPDF (fitz).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PageContent:
    """Holds extracted content for a single page."""

    __slots__ = ("page_number", "text", "headings", "metadata")

    def __init__(
        self,
        page_number: int,
        text: str,
        headings: list[str],
        metadata: dict[str, Any],
    ) -> None:
        self.page_number = page_number
        self.text = text
        self.headings = headings
        self.metadata = metadata


class PDFLoader:
    """
    Loads a PDF document and extracts:
    - Full document metadata (title, author, subject, keywords)
    - Per-page text content
    - Heading detection via font-size heuristics
    - Cleaned, paragraph-aware text
    """

    # Minimum font size (pt) to consider a span a heading
    HEADING_FONT_THRESHOLD_MULTIPLIER = 1.15

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {self.file_path.suffix}")

    def load(self) -> dict[str, Any]:
        """
        Load the PDF and return a structured document dict.

        Returns:
            {
                "title": str,
                "author": str,
                "file_path": str,
                "total_pages": int,
                "metadata": dict,
                "pages": [ {"page": int, "text": str, "headings": [...]} ]
            }
        """
        logger.info("Loading PDF: %s", self.file_path)
        doc = fitz.open(str(self.file_path))

        try:
            raw_meta = doc.metadata or {}
            title = (
                raw_meta.get("title", "").strip()
                or self.file_path.stem.replace("_", " ").replace("-", " ").title()
            )
            author = raw_meta.get("author", "").strip()

            # Calculate body font size mode to identify headings
            body_font_size = self._estimate_body_font_size(doc)

            pages_content: list[dict[str, Any]] = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_data = self._extract_page(page, page_num + 1, body_font_size)
                if page_data["text"].strip():  # Skip blank pages
                    pages_content.append(page_data)

            return {
                "title": title,
                "author": author,
                "file_path": str(self.file_path),
                "total_pages": len(doc),
                "metadata": {
                    "subject": raw_meta.get("subject", ""),
                    "keywords": raw_meta.get("keywords", ""),
                    "creator": raw_meta.get("creator", ""),
                    "producer": raw_meta.get("producer", ""),
                    "format": raw_meta.get("format", ""),
                },
                "pages": pages_content,
            }
        finally:
            doc.close()

    def _estimate_body_font_size(self, doc: fitz.Document) -> float:
        """
        Compute the most common font size across the first 5 pages.
        This heuristic identifies the body text size.
        """
        size_freq: dict[float, int] = {}
        sample_pages = min(5, len(doc))

        for page_num in range(sample_pages):
            page = doc[page_num]
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            for block in blocks:
                if block.get("type") != 0:  # text block
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = round(span.get("size", 0.0), 1)
                        if size > 0:
                            size_freq[size] = size_freq.get(size, 0) + len(span.get("text", ""))

        if not size_freq:
            return 12.0
        return max(size_freq, key=size_freq.get)

    def _extract_page(
        self, page: fitz.Page, page_number: int, body_font_size: float
    ) -> dict[str, Any]:
        """Extract text and headings from a single page."""
        heading_threshold = body_font_size * self.HEADING_FONT_THRESHOLD_MULTIPLIER
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        paragraphs: list[str] = []
        headings: list[str] = []
        current_section = ""

        for block in blocks:
            if block.get("type") != 0:
                continue
            block_lines: list[str] = []
            is_heading_block = False

            for line in block.get("lines", []):
                line_text_parts: list[str] = []
                line_is_heading = False

                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if not span_text:
                        continue
                    span_size = span.get("size", 0.0)
                    span_flags = span.get("flags", 0)
                    is_bold = bool(span_flags & 2**4)

                    if span_size >= heading_threshold or (is_bold and span_size >= body_font_size):
                        line_is_heading = True
                    line_text_parts.append(span_text)

                line_text = " ".join(line_text_parts).strip()
                if not line_text:
                    continue

                block_lines.append(line_text)
                if line_is_heading:
                    is_heading_block = True

            block_text = "\n".join(block_lines).strip()
            if not block_text:
                continue

            if is_heading_block and len(block_text) < 200:
                headings.append(block_text)
                current_section = block_text
                paragraphs.append(f"\n[SECTION: {block_text}]\n")
            else:
                paragraphs.append(block_text)

        full_text = self._clean_text("\n\n".join(paragraphs))

        return {
            "page": page_number,
            "text": full_text,
            "headings": headings,
            "section": headings[-1] if headings else "",
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove hyphenation artifacts, normalize whitespace, strip control chars."""
        # Re-join hyphenated words split across lines
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # Collapse multiple blank lines into a single one
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove non-printable control characters (except newline/tab)
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
        # Normalise unicode whitespace
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
