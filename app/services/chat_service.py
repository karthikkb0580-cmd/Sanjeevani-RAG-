"""
Module 11: Chat Service
app/services/chat_service.py

Orchestrates the full RAG chat pipeline:
  Question → Embed → Retrieve → Rerank → Build Prompt → LLM → Return
"""

from __future__ import annotations

import logging
import time

from app.config.settings import get_settings
from app.embeddings.embedding_service import get_embedding_provider
from app.llm.openai_client import get_llm_client
from app.prompts.prompt_builder import PromptBuilder
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    Citation,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

NOT_FOUND_ANSWER = (
    "I could not find relevant information in the indexed documents."
)


class ChatService:
    """
    Full RAG chat pipeline:

        Question
            ↓ Generate Query Embedding
            ↓ Search Qdrant (Top-K / MMR)
            ↓ Retrieve Top K Chunks
            ↓ Re-rank Results
            ↓ Build Prompt
            ↓ Send to LLM
            ↓ Return Answer + Citations
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._retriever = Retriever()
        self._reranker = Reranker()
        self._prompt_builder = PromptBuilder()
        self._llm = get_llm_client()
        self._embedder = get_embedding_provider()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Answer a user's question using RAG.

        Args:
            request: ChatRequest containing question and retrieval parameters.

        Returns:
            ChatResponse with answer, citations, chunk details, and timing.
        """
        start = time.perf_counter()
        question = request.question

        logger.info("Chat query: '%s…'", question[:80])

        # ── Step 1: Retrieve ─────────────────────────────────────────────────
        chunks: list[RetrievedChunk] = await self._retriever.retrieve(
            query=question,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            document_ids=request.document_ids,
            use_mmr=request.use_mmr,
            mmr_lambda=request.mmr_lambda,
        )

        logger.info("Retrieved %d chunks", len(chunks))

        # ── Step 2: Rerank ───────────────────────────────────────────────────
        if chunks:
            ranked_chunks = self._reranker.rerank(query=question, chunks=chunks)
        else:
            ranked_chunks = []

        # ── Step 3: Build Prompt ─────────────────────────────────────────────
        built_prompt = self._prompt_builder.build(
            question=question,
            chunks=ranked_chunks,
        )

        # ── Step 4: LLM Completion ───────────────────────────────────────────
        if ranked_chunks:
            logger.debug(
                "Sending prompt to LLM (%d chunks, ~%d context tokens)",
                built_prompt.context_chunks_used,
                built_prompt.total_context_tokens,
            )
            llm_response = await self._llm.complete(built_prompt)
            answer = llm_response.content
        else:
            logger.warning("No chunks retrieved – returning not-found answer")
            answer = NOT_FOUND_ANSWER
            llm_response = None

        # ── Step 5: Build Citations ───────────────────────────────────────────
        citations = self._build_citations(ranked_chunks)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Chat completed in %.0f ms | %d citations | model=%s",
            elapsed_ms,
            len(citations),
            self._llm.model_name,
        )

        return ChatResponse(
            question=question,
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,          # Full retrieval set for transparency
            total_chunks_retrieved=len(chunks),
            processing_time_ms=round(elapsed_ms, 2),
            llm_model=self._llm.model_name,
            embedding_model=self._embedder.model_name,
        )

    async def retrieve_only(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        Perform retrieval without calling the LLM.
        Useful for testing / debugging the retrieval pipeline.
        """
        start = time.perf_counter()

        chunks = await self._retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            document_ids=request.document_ids,
            use_mmr=request.use_mmr,
            mmr_lambda=request.mmr_lambda,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RetrievalResponse(
            query=request.query,
            chunks=chunks,
            total_retrieved=len(chunks),
            processing_time_ms=round(elapsed_ms, 2),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
        """Convert retrieved chunks into Citation objects for the response."""
        citations: list[Citation] = []
        seen: set[str] = set()

        for chunk in chunks:
            # Deduplicate citations by (document_id, page, section)
            key = f"{chunk.document_id}::{chunk.page}::{chunk.section}"
            if key in seen:
                continue
            seen.add(key)

            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    title=chunk.title,
                    page=chunk.page,
                    section=chunk.section,
                    chunk_text=chunk.chunk_text,
                    similarity_score=chunk.similarity_score,
                )
            )

        return citations
