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
        Answer a user's question using Agentic RAG.

        Args:
            request: ChatRequest containing question and retrieval parameters.

        Returns:
            ChatResponse with answer, citations, chunk details, and timing.
        """
        start = time.perf_counter()
        question = request.question

        logger.info("Chat query: '%s…'", question[:80])

        # ── Step 1: Agentic RAG loop ─────────────────────────────────────────
        max_iterations = 3
        retrieved_chunks: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()
        queries_run: list[str] = []
        current_query = question

        # Check if the database has any documents to retrieve from.
        # If it is empty, we can skip the agentic retrieval loop entirely to save latency.
        db_empty = False
        try:
            from unittest.mock import Mock
            repo = getattr(self._retriever, "_repository", None)
            if repo and hasattr(repo, "is_empty") and not isinstance(repo, Mock):
                db_empty = await repo.is_empty()
        except Exception as e:
            logger.warning("Error checking if DB is empty: %s", e)

        if db_empty:
            logger.info("Database is empty - skipping agentic RAG loop")
        else:
            for iteration in range(1, max_iterations + 1):
                logger.info(
                    "Agentic RAG Iteration %d/%d: Querying for '%s…'",
                    iteration,
                    max_iterations,
                    current_query[:60],
                )

                # Retrieve chunks for current_query
                chunks = await self._retriever.retrieve(
                    query=current_query,
                    top_k=request.top_k,
                    similarity_threshold=request.similarity_threshold,
                    document_ids=request.document_ids,
                    use_mmr=request.use_mmr,
                    mmr_lambda=request.mmr_lambda,
                )

                new_chunks_added = 0
                for chunk in chunks:
                    if chunk.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk.chunk_id)
                        retrieved_chunks.append(chunk)
                        new_chunks_added += 1

                queries_run.append(current_query)
                logger.info(
                    "Iteration %d retrieved %d chunks (%d new)",
                    iteration,
                    len(chunks),
                    new_chunks_added,
                )

                if new_chunks_added == 0 and len(retrieved_chunks) > 0:
                    logger.info("No new chunks found in iteration %d. Stopping loop.", iteration)
                    break

                if iteration == max_iterations:
                    break

                # Evaluate if more info is needed
                decision = await self._evaluate_need_for_more_info(
                    question=question,
                    queries_run=queries_run,
                    retrieved_chunks=retrieved_chunks,
                )

                if decision.get("action") == "answer":
                    logger.info(
                        "LLM decided we have sufficient information. Stopping loop. Reasoning: %s",
                        decision.get("reasoning"),
                    )
                    break
                elif decision.get("action") == "search" and decision.get("query"):
                    next_query = decision["query"].strip()
                    if next_query in queries_run:
                        logger.info(
                            "LLM suggested query '%s' which was already run. Stopping loop.",
                            next_query,
                        )
                        break
                    current_query = next_query
                else:
                    logger.info("Invalid action or no query provided. Stopping loop.")
                    break

        # ── Step 2: Rerank ───────────────────────────────────────────────────
        if retrieved_chunks:
            ranked_chunks = self._reranker.rerank(query=question, chunks=retrieved_chunks)
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
            logger.info("No chunks retrieved – calling LLM with fallback prompt")
            fallback_system_message = (
                "You are Sanjeevani, an expert scientific research assistant. "
                "No matching documents were found in the indexed knowledge base for your query. "
                "Please answer the user's question using your general scientific knowledge. "
                "You MUST begin your answer by explicitly stating: "
                "'No relevant documents were found in the indexed knowledge base. "
                "The following response is based on general scientific knowledge:' "
                "and then proceed to answer the question accurately and professionally."
            )
            from app.prompts.prompt_builder import BuiltPrompt
            fallback_prompt = BuiltPrompt(
                system_message=fallback_system_message,
                user_message=question,
                context_chunks_used=0,
                total_context_tokens=0,
            )
            try:
                llm_response = await self._llm.complete(fallback_prompt)
                answer = llm_response.content
            except Exception as exc:
                logger.error("Failed to complete fallback LLM call: %s", exc)
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
            retrieved_chunks=retrieved_chunks,  # Full retrieval set for transparency
            total_chunks_retrieved=len(retrieved_chunks),
            processing_time_ms=round(elapsed_ms, 2),
            llm_model=self._llm.model_name,
            embedding_model=self._embedder.model_name,
        )

    async def _evaluate_need_for_more_info(
        self,
        question: str,
        queries_run: list[str],
        retrieved_chunks: list[RetrievedChunk],
    ) -> dict:
        """
        Ask the LLM if the retrieved context is sufficient to answer the question
        or if we need to execute another search query.
        """
        import json
        import re
        from app.prompts.prompt_builder import BuiltPrompt

        # Format the retrieved chunks for the LLM
        context_parts = []
        for i, c in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"Source [{i}]: {c.title} (Page {c.page}, Section {c.section})\n"
                f"Content: {c.chunk_text}"
            )
        context_str = "\n\n".join(context_parts) if context_parts else "No context retrieved yet."

        system_prompt = (
            "You are an intelligent Agentic RAG Router.\n"
            "Your task is to evaluate whether the retrieved scientific passages (if any) "
            "contain sufficient information to fully answer the user's research question.\n"
            "If the information is sufficient, or if further searching is unlikely to yield better results, "
            "you should choose the 'answer' action.\n"
            "If key information is missing (e.g. details, specific metrics, synthesis steps, or definitions) "
            "and could be found in other parts of the literature, you should choose the 'search' action "
            "and generate a refined search query.\n\n"
            "You MUST reply with ONLY a JSON object (no markdown formatting, no explanation, no backticks):\n"
            "{\n"
            "  \"action\": \"search\" | \"answer\",\n"
            "  \"reasoning\": \"A short explanation of what info is present or missing\",\n"
            "  \"query\": \"A refined search query if action is 'search', otherwise empty string\"\n"
            "}"
        )

        user_message = (
            f"User's Question: {question}\n\n"
            f"Queries already executed: {queries_run}\n\n"
            f"Retrieved Passages:\n{context_str}\n\n"
            "Output your JSON decision:"
        )

        built_prompt = BuiltPrompt(
            system_message=system_prompt,
            user_message=user_message,
            context_chunks_used=len(retrieved_chunks),
            total_context_tokens=0,
        )

        try:
            response = await self._llm.complete(built_prompt)
            raw_text = response.content.strip()

            # Clean up markdown code block wrapping if the LLM includes it
            cleaned = raw_text
            if cleaned.startswith("```"):
                cleaned = re.sub(
                    r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE
                ).strip()

            data = json.loads(cleaned)
            return data
        except Exception as exc:
            logger.warning(
                "Failed to evaluate need for more info: %s. Defaulting to 'answer'. Raw text: %s",
                exc,
                raw_text if "raw_text" in locals() else "",
            )
            return {
                "action": "answer",
                "reasoning": f"Error parsing routing decision: {exc}",
                "query": "",
            }

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
