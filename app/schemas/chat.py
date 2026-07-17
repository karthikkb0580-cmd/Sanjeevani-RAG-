"""
Module 2c: Schemas – Chat
app/schemas/chat.py

Pydantic v2 models for chat requests and responses.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    document_ids: list[str] | None = Field(default=None, description="Filter to specific documents")
    use_mmr: bool = Field(default=False, description="Use MMR diversity re-ranking")
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    page: int
    section: str
    chunk_text: str
    similarity_score: float
    chunk_index: int = 0


class RetrievalResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    total_retrieved: int
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_ids: list[str] | None = Field(
        default=None,
        description="Restrict context to specific document IDs",
    )
    top_k: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    use_mmr: bool = Field(default=True)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    stream: bool = Field(default=False, description="Reserved – streaming not yet implemented")


class Citation(BaseModel):
    document_id: str
    title: str
    page: int
    section: str
    chunk_text: str
    similarity_score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    total_chunks_retrieved: int
    processing_time_ms: float
    llm_model: str
    embedding_model: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
