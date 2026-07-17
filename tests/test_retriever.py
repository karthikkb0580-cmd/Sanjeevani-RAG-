"""tests/test_retriever.py – Unit tests for Retriever + Reranker."""

from __future__ import annotations

import math
import pytest

from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker
from app.schemas.chat import RetrievedChunk


def _make_chunk(
    chunk_id: str = "c1",
    text: str = "Transformer models use attention mechanisms.",
    score: float = 0.85,
    page: int = 1,
    section: str = "Introduction",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        title="Test Paper",
        page=page,
        section=section,
        chunk_text=text,
        similarity_score=score,
    )


# ---------------------------------------------------------------------------
# Reranker tests
# ---------------------------------------------------------------------------

class TestReranker:
    def test_rerank_returns_list(self):
        reranker = Reranker(top_n=3)
        chunks = [_make_chunk(chunk_id=f"c{i}", score=0.8) for i in range(5)]
        result = reranker.rerank("attention mechanism", chunks)
        assert isinstance(result, list)

    def test_rerank_respects_top_n(self):
        reranker = Reranker(top_n=2)
        chunks = [_make_chunk(chunk_id=f"c{i}") for i in range(5)]
        result = reranker.rerank("test query", chunks)
        assert len(result) <= 2

    def test_rerank_empty_returns_empty(self):
        reranker = Reranker(top_n=5)
        result = reranker.rerank("test", [])
        assert result == []

    def test_rerank_higher_score_first(self):
        reranker = Reranker(top_n=5)
        chunks = [
            _make_chunk("c1", text="attention mechanism transformer", score=0.90),
            _make_chunk("c2", text="something unrelated", score=0.50),
            _make_chunk("c3", text="attention heads in transformer", score=0.85),
        ]
        result = reranker.rerank("attention transformer", chunks)
        assert result[0].chunk_id in ("c1", "c3")

    def test_rerank_with_single_chunk(self):
        reranker = Reranker(top_n=5)
        chunk = _make_chunk("c1")
        result = reranker.rerank("test", [chunk])
        assert len(result) == 1
        assert result[0].chunk_id == "c1"


# ---------------------------------------------------------------------------
# Cosine similarity tests (Retriever._cosine_similarity)
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        sim = Retriever._cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        sim = Retriever._cosine_similarity(v1, v2)
        assert abs(sim) < 1e-6

    def test_opposite_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        sim = Retriever._cosine_similarity(v1, v2)
        assert abs(sim - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        v1 = [0.0, 0.0]
        v2 = [1.0, 0.0]
        sim = Retriever._cosine_similarity(v1, v2)
        assert sim == 0.0

    def test_empty_vector_returns_zero(self):
        sim = Retriever._cosine_similarity([], [1.0])
        assert sim == 0.0
