"""
Module 8: Retriever
app/retrieval/retriever.py

Implements semantic search, metadata filtering, Top-K, and MMR retrieval
strategies on top of the QdrantRepository.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from app.config.settings import get_settings
from app.embeddings.embedding_service import get_embedding_provider
from app.schemas.chat import RetrievedChunk
from app.vectordb.repository import QdrantRepository

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieval engine that wraps QdrantRepository and the embedding service.

    Strategies:
    - Standard Top-K semantic search
    - MMR (Maximal Marginal Relevance) for diversity-aware retrieval
    Both strategies support optional metadata filtering by document_id.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._repository = QdrantRepository()
        self._embedder = get_embedding_provider()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        document_ids: list[str] | None = None,
        use_mmr: bool = False,
        mmr_lambda: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a given query.

        Args:
            query: Natural-language question or search query.
            top_k: Number of results to return (default from settings).
            similarity_threshold: Minimum cosine similarity score (0–1).
            document_ids: If provided, restrict search to these documents.
            use_mmr: Use Maximal Marginal Relevance instead of plain Top-K.
            mmr_lambda: MMR balance parameter (0=diversity, 1=relevance).

        Returns:
            Ordered list of RetrievedChunk objects (highest relevance first).
        """
        k = top_k or self._settings.retrieval_top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._settings.retrieval_similarity_threshold
        )
        lambda_val = mmr_lambda if mmr_lambda is not None else self._settings.mmr_lambda

        logger.debug("Retrieving for query (top_k=%d, threshold=%.2f, mmr=%s)", k, threshold, use_mmr)

        # Generate query embedding
        query_vector = await self._embedder.embed_query(query)

        if use_mmr:
            return await self._mmr_retrieve(
                query_vector=query_vector,
                top_k=k,
                threshold=threshold,
                document_ids=document_ids,
                mmr_lambda=lambda_val,
            )
        else:
            return await self._semantic_retrieve(
                query_vector=query_vector,
                top_k=k,
                threshold=threshold,
                document_ids=document_ids,
            )

    async def _semantic_retrieve(
        self,
        query_vector: list[float],
        top_k: int,
        threshold: float,
        document_ids: list[str] | None,
    ) -> list[RetrievedChunk]:
        """Standard cosine-similarity Top-K retrieval."""
        results = await self._repository.search_with_filter(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=threshold,
            document_ids=document_ids,
        )
        logger.debug("Semantic search returned %d results", len(results))
        return results

    async def _mmr_retrieve(
        self,
        query_vector: list[float],
        top_k: int,
        threshold: float,
        document_ids: list[str] | None,
        mmr_lambda: float,
    ) -> list[RetrievedChunk]:
        """
        Maximal Marginal Relevance (MMR) retrieval.

        Fetches 3× top_k candidates, then iteratively selects the chunk
        that maximises:
            MMR = λ · sim(query, chunk) − (1−λ) · max_sim(chunk, selected)

        This balances relevance and diversity to reduce redundancy.
        """
        candidate_k = min(top_k * 3, 50)

        candidate_pairs = await self._repository.search_with_vectors(
            query_vector=query_vector,
            top_k=candidate_k,
            score_threshold=threshold,
            document_ids=document_ids,
        )

        if not candidate_pairs:
            logger.debug("MMR: no candidates retrieved")
            return []

        selected: list[RetrievedChunk] = []
        selected_vectors: list[list[float]] = []
        remaining = list(candidate_pairs)

        while remaining and len(selected) < top_k:
            best_chunk: RetrievedChunk | None = None
            best_score = float("-inf")
            best_vec: list[float] = []
            best_idx = -1

            for idx, (chunk, vec) in enumerate(remaining):
                relevance = chunk.similarity_score

                if selected_vectors:
                    max_sim = max(
                        self._cosine_similarity(vec, sel_vec)
                        for sel_vec in selected_vectors
                    )
                else:
                    max_sim = 0.0

                mmr_score = mmr_lambda * relevance - (1.0 - mmr_lambda) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_chunk = chunk
                    best_vec = vec
                    best_idx = idx

            if best_chunk is not None:
                selected.append(best_chunk)
                selected_vectors.append(best_vec)
                remaining.pop(best_idx)

        logger.debug("MMR retrieval selected %d / %d candidates", len(selected), candidate_k)
        return selected

    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not v1 or not v2:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)
