"""
Module 4c: Document Loaders – DOCX
app/loaders/docx_loader.py

Loads Microsoft Word documents (.docx) while preserving
headings (via paragraph styles), paragraphs, and page metadata.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from docx import Document
from docx.oxml.ns import qn

logger = logging.getLogger(__name__)

# Word heading style prefixes
HEADING_STYLE_PREFIXES = ("heading", "title", "subtitle")


class DOCXLoader:
    """
    Loads a .docx document using python-docx and returns a structured
    document dict compatible with the PDF/TXT loader output format.

    Preserves:
    - Document metadata (title, author, subject)
    - Paragraph text
    - Heading styles (Heading 1–9, Title, Subtitle)
    - Approximate page numbers via page-break detection
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {self.file_path}")
        if self.file_path.suffix.lower() not in (".docx", ".doc"):
            raise ValueError(f"Expected a .docx file, got: {self.file_path.suffix}")

    def load(self) -> dict[str, Any]:
        """
        Load the DOCX and return a structured document dict.
        """
        logger.info("Loading DOCX: %s", self.file_path)
        doc = Document(str(self.file_path))

        core_props = doc.core_properties
        title = (
            (core_props.title or "").strip()
            or self.file_path.stem.replace("_", " ").replace("-", " ").title()
        )
        author = (core_props.author or "").strip()

        pages = self._extract_pages(doc)

        return {
            "title": title,
            "author": author,
            "file_path": str(self.file_path),
            "total_pages": len(pages),
            "metadata": {
                "subject": core_props.subject or "",
                "keywords": core_props.keywords or "",
                "created": str(core_props.created or ""),
                "modified": str(core_props.modified or ""),
                "format": "DOCX",
            },
            "pages": pages,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_pages(self, doc: Document) -> list[dict[str, Any]]:
        """
        Split document paragraphs into logical pages.

        Strategy:
        - Detect explicit page breaks (w:lastRenderedPageBreak / w:pageBreak)
        - Fall back to every 40 non-empty paragraphs as a page boundary.
        """
        pages: list[dict[str, Any]] = []
        current_blocks: list[str] = []
        current_headings: list[str] = []
        page_number = 1
        para_count = 0

        def flush_page() -> None:
            nonlocal page_number, current_blocks, current_headings, para_count
            text = "\n\n".join(current_blocks).strip()
            if text:
                pages.append({
                    "page": page_number,
                    "text": self._clean_text(text),
                    "headings": list(current_headings),
                    "section": current_headings[-1] if current_headings else "",
                })
                page_number += 1
            current_blocks = []
            current_headings = []
            para_count = 0

        for para in doc.paragraphs:
            # Detect page break
            if self._has_page_break(para) and current_blocks:
                flush_page()
                continue

            text = para.text.strip()
            if not text:
                continue

            style_name = (para.style.name or "").lower()
            is_heading = any(style_name.startswith(h) for h in HEADING_STYLE_PREFIXES)

            if is_heading:
                current_headings.append(text)
                current_blocks.append(f"\n[SECTION: {text}]\n")
            else:
                current_blocks.append(text)
                para_count += 1

            if para_count >= 40:
                flush_page()

        # Flush remaining content
        if current_blocks:
            flush_page()

        return pages if pages else [{"page": 1, "text": "", "headings": [], "section": ""}]

    @staticmethod
    def _has_page_break(para: Any) -> bool:
        """Check if a paragraph contains a Word page break run."""
        for run in para.runs:
            xml = run._element.xml
            if "w:lastRenderedPageBreak" in xml or 'w:type="page"' in xml:
                return True
        return False

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise whitespace and remove control characters."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
        return text.strip()
