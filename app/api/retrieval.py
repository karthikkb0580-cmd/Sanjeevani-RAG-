"""
Module 12c: API – Retrieval
app/api/retrieval.py

POST /retrieve – Pure semantic retrieval without LLM generation
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.chat import RetrievalRequest, RetrievalResponse
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Retrieval"])


def _get_chat_service() -> ChatService:
    return ChatService()


@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    summary="Semantic document retrieval",
    description=(
        "Retrieve the most relevant document chunks for a query using "
        "semantic search and optional MMR diversity re-ranking. "
        "Does NOT call the LLM."
    ),
)
async def retrieve(
    request: RetrievalRequest,
    service: ChatService = Depends(_get_chat_service),
) -> RetrievalResponse:
    """
    Retrieve relevant chunks without generating an LLM answer.

    Example request:
    ```json
    {
        "query": "What are the limitations of transformer models?",
        "top_k": 5,
        "similarity_threshold": 0.65,
        "use_mmr": true,
        "mmr_lambda": 0.5
    }
    ```

    Example response:
    ```json
    {
        "query": "What are the limitations of transformer models?",
        "chunks": [
            {
                "chunk_id": "...",
                "document_id": "...",
                "title": "Attention Is All You Need",
                "page": 7,
                "section": "Limitations",
                "chunk_text": "...",
                "similarity_score": 0.89,
                "chunk_index": 42
            }
        ],
        "total_retrieved": 5,
        "processing_time_ms": 142.3
    }
    ```
    """
    try:
        return await service.retrieve_only(request)
    except Exception as exc:
        logger.exception("Retrieval failed for query: '%s'", request.query[:80])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        )
