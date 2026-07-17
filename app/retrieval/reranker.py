"""
Module 8b: Re-ranker
app/retrieval/reranker.py

Cross-encoder style re-ranking of retrieved chunks.
Currently implements score-normalisation and top-N selection.
Designed to plug in a neural cross-encoder (e.g. BGE-reranker) when available.
"""

from __future__ import annotations

import logging
import math

from app.config.settings import get_settings
from app.schemas.chat import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """
    Post-retrieval re-ranking of candidate chunks.

    Strategy (lightweight, no external model required):
    1. Normalise similarity scores to [0, 1]
    2. Boost chunks that contain exact query term overlaps
    3. Select top_n from the reranked list

    Plug-in point:
    - Override `_compute_cross_score` with a neural cross-encoder call
      (e.g. sentence-transformers CrossEncoder) when you have the infrastructure.
    """

    def __init__(self, top_n: int | None = None) -> None:
        settings = get_settings()
        self.top_n = top_n or settings.reranker_top_n

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Re-rank chunks and return the top_n results.

        Args:
            query: The original user question.
            chunks: Candidate chunks from the retriever.

        Returns:
            Re-ranked list of up to top_n RetrievedChunks.
        """
        if not chunks:
            return []

        query_terms = set(query.lower().split())

        # Score each chunk
        scored: list[tuple[float, RetrievedChunk]] = []
        for chunk in chunks:
            score = self._compute_rerank_score(query_terms, chunk)
            scored.append((score, chunk))

        # Sort descending by combined score
        scored.sort(key=lambda x: x[0], reverse=True)

        top = [chunk for _, chunk in scored[: self.top_n]]
        logger.debug("Reranker selected %d / %d chunks", len(top), len(chunks))
        return top

    def _compute_rerank_score(
        self,
        query_terms: set[str],
        chunk: RetrievedChunk,
    ) -> float:
        """
        Compute a combined reranking score.

        Formula:
            score = 0.7 × similarity_score + 0.3 × term_overlap_ratio

        Override this method to integrate a neural cross-encoder.
        """
        similarity = chunk.similarity_score

        # Term overlap ratio (Jaccard-style)
        chunk_terms = set(chunk.chunk_text.lower().split())
        if chunk_terms:
            overlap = len(query_terms & chunk_terms) / len(query_terms | chunk_terms)
        else:
            overlap = 0.0

        return 0.7 * similarity + 0.3 * overlap
