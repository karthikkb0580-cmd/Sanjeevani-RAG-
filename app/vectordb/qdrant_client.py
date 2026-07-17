"""
Module 3: Qdrant Integration – Client
app/vectordb/qdrant_client.py

Manages the AsyncQdrantClient lifecycle, collection creation, and health checks.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class QdrantClientManager:
    """
    Singleton wrapper around AsyncQdrantClient.

    Responsibilities:
    - Create / verify the research_documents collection on startup.
    - Expose the raw client for use by the repository layer.
    - Provide health-check utility.
    """

    def __init__(self) -> None:
        self._client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        """Initialise the async Qdrant client and ensure collection exists."""
        settings = get_settings()

        kwargs: dict = {
            "host": settings.qdrant_host,
            "port": settings.qdrant_port,
            "https": settings.qdrant_use_https,
            "timeout": 30,
            "prefer_grpc": False,
        }
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key

        self._client = AsyncQdrantClient(**kwargs)
        logger.info(
            "Qdrant client connected to %s:%s",
            settings.qdrant_host,
            settings.qdrant_port,
        )
        await self._ensure_collection()

    async def disconnect(self) -> None:
        """Close the async Qdrant client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Qdrant client disconnected")

    async def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        settings = get_settings()
        collection_name = settings.qdrant_collection_name
        vector_size = settings.embedding_dimensions

        try:
            exists = await self._client.collection_exists(collection_name)
        except Exception as exc:
            logger.error("Failed to check collection existence: %s", exc)
            raise

        if exists:
            logger.info("Collection '%s' already exists – skipping creation", collection_name)
            return

        logger.info(
            "Creating Qdrant collection '%s' with vector_size=%d",
            collection_name,
            vector_size,
        )
        await self._client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=vector_size,
                distance=qdrant_models.Distance.COSINE,
                on_disk=False,
            ),
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                indexing_threshold=20_000,
                memmap_threshold=50_000,
            ),
            hnsw_config=qdrant_models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10_000,
            ),
        )

        # Create payload indexes for efficient metadata filtering
        await self._create_payload_indexes(collection_name)
        logger.info("Collection '%s' created successfully", collection_name)

    async def _create_payload_indexes(self, collection_name: str) -> None:
        """Create keyword/integer indexes on frequently filtered payload fields."""
        indexed_fields: list[tuple[str, qdrant_models.PayloadSchemaType]] = [
            ("document_id", qdrant_models.PayloadSchemaType.KEYWORD),
            ("title", qdrant_models.PayloadSchemaType.KEYWORD),
            ("page", qdrant_models.PayloadSchemaType.INTEGER),
            ("section", qdrant_models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, field_type in indexed_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
                logger.debug("Payload index created on field '%s'", field_name)
            except UnexpectedResponse as exc:
                # Index may already exist; log and continue
                logger.debug("Payload index '%s' may already exist: %s", field_name, exc)

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantClientManager.connect() has not been called yet")
        return self._client

    async def health_check(self) -> dict:
        """Return Qdrant cluster health information."""
        try:
            info = await self._client.get_collections()
            collection_names = [c.name for c in info.collections]
            return {
                "status": "healthy",
                "collections": collection_names,
            }
        except Exception as exc:
            return {"status": "unhealthy", "error": str(exc)}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_qdrant_manager = QdrantClientManager()


def get_qdrant_manager() -> QdrantClientManager:
    return _qdrant_manager
