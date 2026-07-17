"""
Module 2b: Schemas – Document
app/schemas/document.py

Pydantic v2 models for document upload, indexing, and metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """Metadata stored alongside each chunk in Qdrant."""
    document_id: str
    title: str
    page: int = 0
    section: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentChunk(BaseModel):
    """Represents a single text chunk with its metadata and optional embedding."""
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    title: str
    page: int = 0
    section: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    text: str
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# API Request / Response
# ---------------------------------------------------------------------------

class IndexDocumentRequest(BaseModel):
    """Optional overrides when indexing a single document."""
    title: str | None = Field(default=None, description="Override auto-detected title")
    chunk_size: int | None = Field(default=None, ge=100, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=500)


class IndexDocumentResponse(BaseModel):
    document_id: str
    title: str
    total_chunks: int
    pages: int
    processing_time_ms: float
    status: str = "indexed"
    message: str = "Document indexed successfully"


class BatchIndexResponse(BaseModel):
    total_files: int
    successful: int
    failed: int
    results: list[IndexDocumentResponse]
    errors: list[dict[str, Any]] = Field(default_factory=list)


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted_chunks: int
    status: str = "deleted"
    message: str = "Document and all associated chunks deleted"
