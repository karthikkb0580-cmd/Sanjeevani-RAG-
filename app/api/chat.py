"""
Module 12d: API – Chat
app/api/chat.py

POST /chat – Full RAG chat endpoint
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


def _get_chat_service() -> ChatService:
    return ChatService()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with indexed research papers",
    description=(
        "Ask a question about indexed research papers. The service retrieves "
        "the most relevant document chunks, re-ranks them, builds a grounded "
        "prompt, and generates an answer using the configured LLM. "
        "Returns the answer with full citations."
    ),
)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(_get_chat_service),
) -> ChatResponse:
    """
    Full RAG chat endpoint.

    Example request:
    ```json
    {
        "question": "What are the key contributions of the transformer architecture?",
        "top_k": 10,
        "similarity_threshold": 0.65,
        "use_mmr": true,
        "mmr_lambda": 0.5
    }
    ```

    Example response:
    ```json
    {
        "question": "What are the key contributions of the transformer architecture?",
        "answer": "According to 'Attention Is All You Need' (page 1, section Introduction): ...",
        "citations": [
            {
                "document_id": "...",
                "title": "Attention Is All You Need",
                "page": 1,
                "section": "Introduction",
                "chunk_text": "...",
                "similarity_score": 0.92
            }
        ],
        "retrieved_chunks": [...],
        "total_chunks_retrieved": 8,
        "processing_time_ms": 2341.7,
        "llm_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
        "timestamp": "2024-01-15T10:30:00Z"
    }
    ```
    """
    try:
        return await service.chat(request)
    except Exception as exc:
        logger.exception("Chat failed for question: '%s'", request.question[:80])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat pipeline failed: {exc}",
        )
