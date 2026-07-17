"""
app/vision/document_extractor.py

In-memory document text extractor for the vision multipart endpoint.

Supports:
  - PDF  (.pdf)   via PyMuPDF  (fitz) — already in requirements.txt
  - DOCX (.docx)  via python-docx    — already in requirements.txt

Works entirely from raw bytes so no temp files are written to disk.

Returns a DocumentExtractionResult with:
  - full_text   : concatenated text of all pages (first MAX_CHARS chars)
  - title       : document title from metadata or filename
  - author      : document author if available
  - page_count  : total pages
  - headings    : list of detected section headings
  - research_query: summary string suitable as a RAG query
"""

from __future__ import annotations

import io
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Max characters of extracted text to forward to the orchestrator
# (~8 000 tokens ≈ comfortable context for Nemotron)
MAX_CHARS = 32_000


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class DocumentExtractionResult(BaseModel):
    """Structured text content extracted from a PDF or DOCX file."""

    file_type:      Literal["pdf", "docx", "unknown"] = Field(default="unknown")
    title:          str = Field(default="")
    author:         str = Field(default="")
    page_count:     int = Field(default=0)
    headings:       list[str] = Field(default_factory=list)
    full_text:      str = Field(default="", description="Extracted text (truncated to MAX_CHARS)")
    research_query: str = Field(default="", description="Title + first 500 chars of text")
    warnings:       list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# PDF extraction (bytes → DocumentExtractionResult)
# ---------------------------------------------------------------------------

def extract_pdf(file_bytes: bytes, filename: str = "document.pdf") -> DocumentExtractionResult:
    """Extract text from a PDF supplied as raw bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return DocumentExtractionResult(
            file_type="pdf",
            warnings=["PyMuPDF (fitz) is not installed. Install pymupdf."],
        )

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        return DocumentExtractionResult(
            file_type="pdf",
            warnings=[f"Failed to open PDF: {exc}"],
        )

    try:
        raw_meta = doc.metadata or {}
        title  = (raw_meta.get("title", "").strip()
                  or re.sub(r"[_-]", " ", re.sub(r"\.pdf$", "", filename, flags=re.I)).title())
        author = raw_meta.get("author", "").strip()

        all_text_parts: list[str] = []
        all_headings:   list[str] = []

        # Font-size heuristic for heading detection
        size_freq: dict[float, int] = {}
        sample = min(5, len(doc))
        for p in range(sample):
            for b in doc[p].get_text("dict")["blocks"]:
                if b.get("type") != 0:
                    continue
                for ln in b.get("lines", []):
                    for sp in ln.get("spans", []):
                        sz = round(sp.get("size", 0.0), 1)
                        if sz > 0:
                            size_freq[sz] = size_freq.get(sz, 0) + len(sp.get("text", ""))
        body_sz   = max(size_freq, key=size_freq.get) if size_freq else 12.0
        head_thr  = body_sz * 1.15

        for page_num in range(len(doc)):
            page_text_parts: list[str] = []
            for block in doc[page_num].get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                block_lines: list[str] = []
                is_heading = False
                for ln in block.get("lines", []):
                    line_parts: list[str] = []
                    for sp in ln.get("spans", []):
                        txt = sp.get("text", "").strip()
                        if not txt:
                            continue
                        sz    = sp.get("size", 0.0)
                        flags = sp.get("flags", 0)
                        bold  = bool(flags & (2 ** 4))
                        if sz >= head_thr or (bold and sz >= body_sz):
                            is_heading = True
                        line_parts.append(txt)
                    joined = " ".join(line_parts).strip()
                    if joined:
                        block_lines.append(joined)
                block_str = "\n".join(block_lines).strip()
                if not block_str:
                    continue
                if is_heading and len(block_str) < 200:
                    all_headings.append(block_str)
                    page_text_parts.append(f"\n[SECTION: {block_str}]\n")
                else:
                    page_text_parts.append(block_str)
            all_text_parts.append("\n\n".join(page_text_parts))

        full_text = _clean("\n\n".join(all_text_parts))[:MAX_CHARS]
        research_query = _build_query(title, full_text)

        return DocumentExtractionResult(
            file_type="pdf",
            title=title,
            author=author,
            page_count=len(doc),
            headings=all_headings[:30],
            full_text=full_text,
            research_query=research_query,
        )
    except Exception as exc:
        logger.exception("PDF extraction failed: %s", exc)
        return DocumentExtractionResult(
            file_type="pdf",
            warnings=[f"Extraction error: {exc}"],
        )
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# DOCX extraction (bytes → DocumentExtractionResult)
# ---------------------------------------------------------------------------

def extract_docx(file_bytes: bytes, filename: str = "document.docx") -> DocumentExtractionResult:
    """Extract text from a DOCX supplied as raw bytes using python-docx."""
    try:
        from docx import Document
    except ImportError:
        return DocumentExtractionResult(
            file_type="docx",
            warnings=["python-docx is not installed. Install python-docx."],
        )

    try:
        doc = Document(io.BytesIO(file_bytes))
    except Exception as exc:
        return DocumentExtractionResult(
            file_type="docx",
            warnings=[f"Failed to open DOCX: {exc}"],
        )

    try:
        core = doc.core_properties
        title  = ((core.title or "").strip()
                  or re.sub(r"[_-]", " ", re.sub(r"\.docx?$", "", filename, flags=re.I)).title())
        author = (core.author or "").strip()

        HEADING_PREFIXES = ("heading", "title", "subtitle")
        all_parts:    list[str] = []
        all_headings: list[str] = []
        page_count    = 1

        for para in doc.paragraphs:
            txt = para.text.strip()
            if not txt:
                continue
            style = (para.style.name or "").lower()
            is_h  = any(style.startswith(p) for p in HEADING_PREFIXES)
            if is_h:
                all_headings.append(txt)
                all_parts.append(f"\n[SECTION: {txt}]\n")
            else:
                all_parts.append(txt)

            # Count page breaks as rough page estimator
            for run in para.runs:
                xml = run._element.xml
                if "w:lastRenderedPageBreak" in xml or 'w:type="page"' in xml:
                    page_count += 1

        full_text = _clean("\n\n".join(all_parts))[:MAX_CHARS]
        research_query = _build_query(title, full_text)

        return DocumentExtractionResult(
            file_type="docx",
            title=title,
            author=author,
            page_count=page_count,
            headings=all_headings[:30],
            full_text=full_text,
            research_query=research_query,
        )
    except Exception as exc:
        logger.exception("DOCX extraction failed: %s", exc)
        return DocumentExtractionResult(
            file_type="docx",
            warnings=[f"Extraction error: {exc}"],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Normalise whitespace and strip control characters."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)       # de-hyphenate
    text = re.sub(r"\n{3,}", "\n\n", text)               # collapse blank lines
    text = re.sub(r"[ \t]+", " ", text)                  # collapse spaces
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
    return text.strip()


def _build_query(title: str, full_text: str) -> str:
    """Build a concise query string for the RAG pipeline."""
    snippet = full_text[:500].replace("\n", " ").strip()
    if title:
        return f"{title} — {snippet}"
    return snippet
