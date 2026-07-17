"""
Module 3b: Qdrant Integration – Repository
app/vectordb/repository.py

All Qdrant read/write operations live here.
No business logic — pure data access.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient, models as qdrant_models

from app.config.settings import get_settings
from app.schemas.document import DocumentChunk
from app.schemas.chat import RetrievedChunk
from app.vectordb.qdrant_client import get_qdrant_manager

logger = logging.getLogger(__name__)


class QdrantRepository:
    """
    Repository layer for all Qdrant operations.

    Methods:
    - upsert_chunks          – Store/update a batch of embedded chunks
    - search_similar         – Cosine similarity search
    - search_with_filter     – Similarity search restricted by payload filter
    - delete_document_chunks – Remove all points belonging to a document_id
    - count_document_chunks  – Count points for a given document_id
    - get_chunk_by_id        – Retrieve a single point by its UUID
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    @property
    def _client(self) -> AsyncQdrantClient:
        return get_qdrant_manager().client

    @property
    def _collection(self) -> str:
        return self._settings.qdrant_collection_name

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        Upsert a list of DocumentChunks into Qdrant.

        Returns the number of points actually upserted.
        """
        if not chunks:
            return 0

        points: list[qdrant_models.PointStruct] = []
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning("Chunk %s has no embedding – skipping", chunk.chunk_id)
                continue

            point = qdrant_models.PointStruct(
                id=chunk.chunk_id,  # Already a valid UUID string
                vector=chunk.embedding,
                payload={
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "page": chunk.page,
                    "section": chunk.section,
                    "chunk_text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "created_at": chunk.created_at.isoformat(),
                },
            )
            points.append(point)

        if not points:
            logger.warning("No valid points to upsert (all chunks missing embeddings)")
            return 0

        # Batch in groups of 100 to avoid payload size limits
        batch_size = 100
        upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._client.upsert(
                collection_name=self._collection,
                points=batch,
                wait=True,
            )
            upserted += len(batch)
            logger.debug("Upserted batch %d/%d (%d points)", i // batch_size + 1, -(-len(points) // batch_size), len(batch))

        logger.info("Upserted %d points for document '%s'", upserted, chunks[0].document_id)
        return upserted

    # ------------------------------------------------------------------
    # Read – similarity search
    # ------------------------------------------------------------------

    async def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Cosine similarity search without payload filter."""
        response = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return [self._point_to_retrieved_chunk(r) for r in response.points]

    async def search_with_filter(
        self,
        query_vector: list[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Cosine similarity search with optional document_id filter."""
        query_filter: qdrant_models.Filter | None = None

        if document_ids:
            query_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchAny(any=document_ids),
                    )
                ]
            )

        response = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return [self._point_to_retrieved_chunk(r) for r in response.points]

    async def search_with_vectors(
        self,
        query_vector: list[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        document_ids: list[str] | None = None,
    ) -> list[tuple[RetrievedChunk, list[float]]]:
        """
        Similarity search that also returns the stored vectors.
        Used by the MMR reranker which needs candidate embeddings.
        """
        query_filter: qdrant_models.Filter | None = None
        if document_ids:
            query_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchAny(any=document_ids),
                    )
                ]
            )

        response = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=True,
        )
        pairs: list[tuple[RetrievedChunk, list[float]]] = []
        for r in response.points:
            chunk = self._point_to_retrieved_chunk(r)
            vec: list[float] = r.vector if isinstance(r.vector, list) else []
            pairs.append((chunk, vec))
        return pairs

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all Qdrant points whose payload.document_id matches.

        Returns the count of deleted points (approximated via pre-delete count).
        """
        count_before = await self.count_document_chunks(document_id)

        await self._client.delete(
            collection_name=self._collection,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="document_id",
                            match=qdrant_models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
        logger.info("Deleted %d points for document_id='%s'", count_before, document_id)
        return count_before

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    async def count_document_chunks(self, document_id: str) -> int:
        """Return the number of stored chunks for a given document_id."""
        result = await self._client.count(
            collection_name=self._collection,
            count_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchValue(value=document_id),
                    )
                ]
            ),
            exact=True,
        )
        return result.count

    async def is_empty(self) -> bool:
        """Check if the collection contains zero points."""
        try:
            info = await self._client.get_collection(self._collection)
            return info.points_count == 0
        except Exception as exc:
            logger.warning("Failed to check if collection is empty: %s", exc)
            return True

    async def get_chunk_by_id(self, chunk_id: str) -> RetrievedChunk | None:
        """Retrieve a single chunk by its UUID point ID."""
        results = await self._client.retrieve(
            collection_name=self._collection,
            ids=[chunk_id],
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        return self._point_to_retrieved_chunk(results[0])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _point_to_retrieved_chunk(point: Any) -> RetrievedChunk:
        """Convert a Qdrant ScoredPoint or Record into a RetrievedChunk."""
        payload = point.payload or {}
        return RetrievedChunk(
            chunk_id=str(point.id),
            document_id=payload.get("document_id", ""),
            title=payload.get("title", ""),
            page=payload.get("page", 0),
            section=payload.get("section", ""),
            chunk_text=payload.get("chunk_text", ""),
            similarity_score=getattr(point, "score", 0.0),
            chunk_index=payload.get("chunk_index", 0),
        )
