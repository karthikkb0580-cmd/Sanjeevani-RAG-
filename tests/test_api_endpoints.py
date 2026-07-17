"""tests/test_api_endpoints.py – Integration tests for chat, retrieve, and documents API endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch, MagicMock

from app.schemas.chat import ChatResponse, RetrievalResponse, RetrievedChunk, Citation
from app.schemas.document import DeleteDocumentResponse


@pytest.mark.asyncio
async def test_chat_endpoint_success(client: AsyncClient):
    mock_chat_response = ChatResponse(
        question="What is a ribosome?",
        answer="A ribosome makes proteins.",
        citations=[
            Citation(
                document_id="doc-123",
                title="Cell Biology",
                page=5,
                section="Translation",
                chunk_text="Ribosomes are the site of protein synthesis.",
                similarity_score=0.89
            )
        ],
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-123",
                title="Cell Biology",
                page=5,
                section="Translation",
                chunk_text="Ribosomes are the site of protein synthesis.",
                similarity_score=0.89
            )
        ],
        total_chunks_retrieved=1,
        processing_time_ms=150.0,
        llm_model="gpt-4o",
        embedding_model="text-embedding-3-small"
    )

    with patch("app.services.chat_service.ChatService.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = mock_chat_response

        response = await client.post(
            "/chat",
            json={
                "question": "What is a ribosome?",
                "top_k": 3,
                "similarity_threshold": 0.6
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is a ribosome?"
        assert data["answer"] == "A ribosome makes proteins."
        assert len(data["citations"]) == 1


@pytest.mark.asyncio
async def test_retrieve_endpoint_success(client: AsyncClient):
    mock_retrieval_response = RetrievalResponse(
        query="mitochondria",
        chunks=[
            RetrievedChunk(
                chunk_id="chunk-2",
                document_id="doc-123",
                title="Cell Biology",
                page=10,
                section="Energy",
                chunk_text="Mitochondria are the powerhouse of the cell.",
                similarity_score=0.95
            )
        ],
        total_retrieved=1,
        processing_time_ms=50.0
    )

    with patch("app.services.chat_service.ChatService.retrieve_only", new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = mock_retrieval_response

        response = await client.post(
            "/retrieve",
            json={
                "query": "mitochondria",
                "top_k": 5
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "mitochondria"
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["section"] == "Energy"


@pytest.mark.asyncio
async def test_delete_document_success(client: AsyncClient):
    with patch("app.services.indexing_service.IndexingService.delete_document", new_callable=AsyncMock) as mock_delete:
        mock_delete.return_value = 12

        response = await client.delete("/documents/doc-123")
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "doc-123"
        assert data["deleted_chunks"] == 12
