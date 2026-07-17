"""
Module 5: Chunking
app/chunking/chunker.py

Recursive Character Text Splitter that produces DocumentChunk objects
from the loaded page content. Preserves section headings and metadata.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import get_settings
from app.schemas.document import DocumentChunk
from app.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Splits raw document pages into overlapping text chunks
    using the RecursiveCharacterTextSplitter with token-aware sizing.

    Hierarchy of separators:
    1. Double newline (paragraph boundary)
    2. Single newline
    3. Sentence-ending punctuation
    4. Spaces
    5. Empty string (last resort)
    """

    SEPARATORS = [
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
        "; ",
        ", ",
        " ",
        "",
    ]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        chunk_min_size: int | None = None,
    ) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.chunk_min_size = chunk_min_size or settings.chunk_min_size

        # Use tiktoken-based length function for accurate token counting
        self._splitter = RecursiveCharacterTextSplitter(
            separators=self.SEPARATORS,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=count_tokens,
            is_separator_regex=False,
            keep_separator=True,
        )
        logger.debug(
            "DocumentChunker initialised: chunk_size=%d, overlap=%d",
            self.chunk_size,
            self.chunk_overlap,
        )

    def chunk_document(
        self,
        document: dict[str, Any],
        document_id: str,
    ) -> list[DocumentChunk]:
        """
        Split an entire document (all pages) into DocumentChunk objects.

        Args:
            document: Output of any loader (pdf/txt/docx).
            document_id: UUID string assigned to this document.

        Returns:
            Flat list of DocumentChunk objects, ordered by page then chunk index.
        """
        title = document.get("title", "Untitled")
        pages: list[dict[str, Any]] = document.get("pages", [])
        all_chunks: list[DocumentChunk] = []

        for page_data in pages:
            page_number: int = page_data.get("page", 0)
            section: str = page_data.get("section", "")
            text: str = page_data.get("text", "").strip()

            if not text or count_tokens(text) < self.chunk_min_size:
                logger.debug("Skipping page %d – too short", page_number)
                continue

            page_chunks = self._split_page(
                text=text,
                document_id=document_id,
                title=title,
                page_number=page_number,
                section=section,
            )
            all_chunks.extend(page_chunks)

        # Assign global chunk indices
        total = len(all_chunks)
        for idx, chunk in enumerate(all_chunks):
            chunk.chunk_index = idx
            chunk.total_chunks = total

        logger.info(
            "Document '%s' split into %d chunks across %d page(s)",
            title,
            total,
            len(pages),
        )
        return all_chunks

    def _split_page(
        self,
        text: str,
        document_id: str,
        title: str,
        page_number: int,
        section: str,
    ) -> list[DocumentChunk]:
        """Split a single page text into chunks."""
        raw_chunks: list[str] = self._splitter.split_text(text)

        chunks: list[DocumentChunk] = []
        for raw in raw_chunks:
            cleaned = raw.strip()
            if not cleaned or count_tokens(cleaned) < self.chunk_min_size:
                continue

            # Detect if a SECTION marker has been carried into this chunk
            chunk_section = self._extract_section_from_text(cleaned) or section

            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    title=title,
                    page=page_number,
                    section=chunk_section,
                    text=cleaned,
                )
            )
        return chunks

    @staticmethod
    def _extract_section_from_text(text: str) -> str:
        """
        If a [SECTION: ...] marker was carried into this chunk,
        extract it as the heading for this chunk.
        """
        import re
        match = re.search(r"\[SECTION:\s*(.+?)\]", text)
        return match.group(1).strip() if match else ""
