"""
Module 7: Indexing Service
app/services/indexing_service.py

Orchestrates the full document indexing pipeline:
  Load → Extract → Clean → Chunk → Embed → Store
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from app.chunking.chunker import DocumentChunker
from app.config.settings import get_settings
from app.embeddings.embedding_service import get_embedding_provider
from app.loaders import load_document
from app.schemas.document import DocumentChunk, IndexDocumentResponse
from app.vectordb.repository import QdrantRepository

logger = logging.getLogger(__name__)


class IndexingService:
    """
    Orchestrates the complete indexing pipeline for research documents.

    Pipeline:
        Document File
            ↓ Load (PDF / TXT / DOCX)
            ↓ Extract Text (with page / heading metadata)
            ↓ Clean Text
            ↓ Chunk (Recursive Character Splitter)
            ↓ Generate Embeddings (batch, async)
            ↓ Store in Qdrant
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._repository = QdrantRepository()
        self._embedder = get_embedding_provider()

    async def index_document(
        self,
        file_path: str | Path,
        document_id: str | None = None,
        title_override: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IndexDocumentResponse:
        """
        Run the full indexing pipeline for a single document.

        Args:
            file_path: Path to the uploaded file.
            document_id: Optional UUID override; generated if not provided.
            title_override: Optional title to use instead of auto-detected one.
            chunk_size: Override default chunk token size.
            chunk_overlap: Override default chunk overlap.

        Returns:
            IndexDocumentResponse with stats and status.
        """
        start = time.perf_counter()
        file_path = Path(file_path)
        doc_id = document_id or str(uuid.uuid4())

        logger.info("Starting indexing pipeline for '%s' (id=%s)", file_path.name, doc_id)

        # ── Step 1: Load document ────────────────────────────────────────────
        document = load_document(file_path)
        title = title_override or document["title"]
        total_pages = document["total_pages"]
        logger.debug("Loaded document '%s' – %d pages", title, total_pages)

        # ── Step 2: Chunk ────────────────────────────────────────────────────
        chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks: list[DocumentChunk] = chunker.chunk_document(document, doc_id)
        if not chunks:
            logger.warning("No usable chunks produced for '%s'", title)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return IndexDocumentResponse(
                document_id=doc_id,
                title=title,
                total_chunks=0,
                pages=total_pages,
                processing_time_ms=round(elapsed_ms, 2),
                status="warning",
                message="No usable text chunks could be extracted from the document",
            )

        logger.info("Produced %d chunks for '%s'", len(chunks), title)

        # ── Step 3: Generate Embeddings ──────────────────────────────────────
        texts = [chunk.text for chunk in chunks]
        logger.debug("Generating embeddings for %d chunks …", len(texts))
        embeddings = await self._embedder.embed_texts(texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            chunk.title = title  # Propagate title override

        logger.debug("Embeddings generated (%d vectors)", len(embeddings))

        # ── Step 4: Store in Qdrant ──────────────────────────────────────────
        upserted = await self._repository.upsert_chunks(chunks)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Indexed '%s': %d/%d chunks stored in %.0f ms",
            title,
            upserted,
            len(chunks),
            elapsed_ms,
        )

        return IndexDocumentResponse(
            document_id=doc_id,
            title=title,
            total_chunks=upserted,
            pages=total_pages,
            processing_time_ms=round(elapsed_ms, 2),
            status="indexed",
            message=f"Successfully indexed {upserted} chunks from {total_pages} pages",
        )

    async def delete_document(self, document_id: str) -> int:
        """
        Remove all Qdrant points for the given document_id.

        Returns:
            Number of deleted chunks.
        """
        logger.info("Deleting document '%s' from vector store", document_id)
        deleted = await self._repository.delete_document_chunks(document_id)
        logger.info("Deleted %d chunks for document '%s'", deleted, document_id)
        return deleted
