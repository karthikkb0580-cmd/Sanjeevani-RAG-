"""
Module 4b: Document Loaders – TXT
app/loaders/txt_loader.py

Loads plain-text files while preserving paragraphs and detecting
section headings using capitalisation / underline heuristics.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TXTLoader:
    """
    Loads a .txt document and returns a structured document dict
    compatible with the PDF loader output format.

    Heading detection heuristics:
    - All-uppercase short lines (≤ 80 chars)
    - Lines followed by ===== or ----- underlines
    - Lines matching "1.", "1.1", "Chapter X", "Section X" patterns
    """

    HEADING_PATTERNS = [
        re.compile(r"^#{1,6}\s+.+$"),                      # Markdown headings
        re.compile(r"^(?:Chapter|Section|Part)\s+\w+", re.IGNORECASE),
        re.compile(r"^\d+(\.\d+)*\s+[A-Z]"),               # Numbered sections
        re.compile(r"^[A-Z][A-Z\s,\-:]{5,79}$"),           # ALL CAPS lines
    ]
    UNDERLINE_PATTERN = re.compile(r"^[=\-]{3,}$")

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"TXT file not found: {self.file_path}")

    def load(self) -> dict[str, Any]:
        """
        Load the TXT file and return a structured document dict.
        """
        logger.info("Loading TXT: %s", self.file_path)

        with self.file_path.open("r", encoding="utf-8", errors="replace") as fh:
            raw_content = fh.read()

        lines = raw_content.splitlines()
        title = self._infer_title(lines)

        pages = self._split_into_pages(lines)

        return {
            "title": title,
            "author": "",
            "file_path": str(self.file_path),
            "total_pages": len(pages),
            "metadata": {"format": "TXT"},
            "pages": pages,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_title(self, lines: list[str]) -> str:
        """Use the first non-empty line as the document title."""
        for line in lines[:20]:
            stripped = line.strip()
            if stripped and len(stripped) > 3:
                # Remove leading # for markdown
                return re.sub(r"^#+\s*", "", stripped)
        return self.file_path.stem.replace("_", " ").replace("-", " ").title()

    def _split_into_pages(self, lines: list[str]) -> list[dict[str, Any]]:
        """
        Treat every ~50 non-empty lines as a logical 'page'.
        Detects headings and builds per-page dicts.
        """
        pages: list[dict[str, Any]] = []
        current_lines: list[str] = []
        current_headings: list[str] = []
        current_section = ""
        page_number = 1
        non_empty_count = 0

        def flush_page() -> None:
            nonlocal page_number, current_lines, current_headings, current_section, non_empty_count
            text = self._build_page_text(current_lines, current_headings)
            if text.strip():
                pages.append({
                    "page": page_number,
                    "text": text,
                    "headings": list(current_headings),
                    "section": current_headings[-1] if current_headings else "",
                })
                page_number += 1
            current_lines = []
            current_headings = []
            non_empty_count = 0

        for idx, line in enumerate(lines):
            # Detect explicit page breaks
            if line.strip() == "\x0c":
                flush_page()
                continue

            stripped = line.strip()
            is_heading = self._is_heading(stripped, lines, idx)

            if is_heading and stripped:
                current_headings.append(stripped)
                current_section = stripped
                current_lines.append(f"\n[SECTION: {stripped}]\n")
            else:
                current_lines.append(line)
                if stripped:
                    non_empty_count += 1

            # Logical page boundary every ~50 content lines
            if non_empty_count >= 50:
                flush_page()

        # Flush remainder
        if current_lines:
            flush_page()

        return pages if pages else [{"page": 1, "text": "\n".join(lines), "headings": [], "section": ""}]

    def _is_heading(self, line: str, all_lines: list[str], idx: int) -> bool:
        """Return True if this line looks like a section heading."""
        if not line or len(line) > 120:
            return False
        # Check next line for underline
        if idx + 1 < len(all_lines):
            next_line = all_lines[idx + 1].strip()
            if self.UNDERLINE_PATTERN.match(next_line):
                return True
        for pattern in self.HEADING_PATTERNS:
            if pattern.match(line):
                return True
        return False

    @staticmethod
    def _build_page_text(lines: list[str], headings: list[str]) -> str:
        """Build clean page text from lines and headings."""
        text = "\n".join(lines)
        # Collapse 3+ blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
