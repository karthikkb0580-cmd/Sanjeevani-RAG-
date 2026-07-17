"""
Module 12a: API – Health
app/api/health.py
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.config.settings import Settings, get_settings
from app.vectordb.qdrant_client import get_qdrant_manager

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Service health check",
    description="Returns the operational status of the RAG service and its dependencies.",
    response_model=dict,
)
async def health_check(settings: Settings = Depends(get_settings)) -> dict:
    """
    Comprehensive health check endpoint.

    Returns service version, Qdrant connectivity, and runtime info.

    Example response:
    ```json
    {
        "status": "healthy",
        "service": "Sanjeevani RAG Service",
        "version": "1.0.0",
        "environment": "development",
        "timestamp": "2024-01-15T10:30:00Z",
        "dependencies": {
            "qdrant": {"status": "healthy", "collections": ["research_documents"]}
        },
        "runtime": {"python": "3.12.0", "platform": "Linux"}
    }
    ```
    """
    manager = get_qdrant_manager()
    qdrant_health = await manager.health_check()

    overall_status = "healthy" if qdrant_health["status"] == "healthy" else "degraded"

    return {
        "status": overall_status,
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": {
            "qdrant": qdrant_health,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.system(),
        },
        "config": {
            "embedding_model": settings.openai_embedding_model,
            "chat_model": settings.openai_chat_model,
            "collection": settings.qdrant_collection_name,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieval_top_k": settings.retrieval_top_k,
        },
    }
