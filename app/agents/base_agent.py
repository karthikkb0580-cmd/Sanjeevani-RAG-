"""
app/agents/base_agent.py

Abstract base class for all Sanjeevani analysis agents.

Design principles:
- Every agent is independently testable and replaceable.
- Each agent performs retrieval BEFORE reasoning.
- Each agent catches its own exceptions and returns an error AgentResult.
- Dependency injection: Retriever and LLM client via constructor.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from app.llm.openai_client import BaseLLMClient, get_llm_client
from app.prompts.prompt_builder import BuiltPrompt
from app.retrieval.retriever import Retriever
from app.schemas.agents import (
    AgentResult,
    Confidence,
    EvidenceItem,
)
from app.schemas.analysis import AnalysisRequest
from app.schemas.chat import RetrievedChunk

logger = logging.getLogger(__name__)

NO_EVIDENCE_MSG = (
    "No supporting evidence was found in the indexed knowledge base."
)


class BaseAgent(ABC):
    """
    Abstract base for all Sanjeevani multi-agents.

    Subclasses must implement:
    - agent_name  : str property
    - run()       : async method returning AgentResult
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self._retriever   = retriever   or Retriever()
        self._llm         = llm_client  or get_llm_client()

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Return the canonical agent name (used in AgentResult and logging)."""

    @abstractmethod
    async def run(self, request: AnalysisRequest) -> AgentResult:
        """
        Execute the agent's full pipeline:
          1. Build retrieval queries
          2. Retrieve evidence
          3. Reason over evidence
          4. Return structured AgentResult

        Never raises — catches all exceptions and returns error state.
        """

    # ------------------------------------------------------------------
    # Shared helpers available to all agents
    # ------------------------------------------------------------------

    async def _retrieve(
        self,
        query: str,
        request: AnalysisRequest,
    ) -> list[RetrievedChunk]:
        """
        Execute retrieval using the agent's shared Retriever instance.
        Uses parameters from the original AnalysisRequest.
        """
        return await self._retriever.retrieve(
            query=query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            use_mmr=request.use_mmr,
            mmr_lambda=request.mmr_lambda,
        )

    async def _retrieve_multi(
        self,
        queries: list[str],
        request: AnalysisRequest,
    ) -> list[RetrievedChunk]:
        """
        Run multiple retrieval queries and merge results (deduplicated).
        Useful for agents that probe different aspects of a molecule.
        """
        import asyncio

        tasks = [self._retrieve(q, request) for q in queries]
        results_nested = await asyncio.gather(*tasks, return_exceptions=True)

        seen_ids: set[str] = set()
        merged: list[RetrievedChunk] = []
        for res in results_nested:
            if isinstance(res, Exception):
                logger.warning("%s: sub-query failed: %s", self.agent_name, res)
                continue
            for chunk in res:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    merged.append(chunk)

        # Re-sort by similarity descending
        merged.sort(key=lambda c: c.similarity_score, reverse=True)
        return merged

    @staticmethod
    def _chunks_to_evidence(chunks: list[RetrievedChunk]) -> list[EvidenceItem]:
        """Convert retrieved chunks into EvidenceItem objects."""
        return [
            EvidenceItem(
                document_id=c.document_id,
                title=c.title,
                page=c.page,
                section=c.section,
                chunk_text=c.chunk_text,
                similarity_score=c.similarity_score,
                evidence_type="retrieved",
            )
            for c in chunks
        ]

    @staticmethod
    def _derive_confidence(chunks: list[RetrievedChunk]) -> Confidence:
        """
        Derive evidence confidence from retrieval quality.

        HIGH   : ≥5 chunks with avg similarity ≥ 0.78
        MEDIUM : ≥2 chunks with avg similarity ≥ 0.65
        LOW    : any chunks found
        NONE   : no chunks retrieved
        """
        if not chunks:
            return Confidence.NONE

        avg_score = sum(c.similarity_score for c in chunks) / len(chunks)
        n = len(chunks)

        if n >= 5 and avg_score >= 0.78:
            return Confidence.HIGH
        if n >= 2 and avg_score >= 0.65:
            return Confidence.MEDIUM
        return Confidence.LOW

    def _build_simple_prompt(
        self,
        system: str,
        context: str,
        question: str,
    ) -> BuiltPrompt:
        """Build a BuiltPrompt directly (without the PromptBuilder token caps)."""
        from app.prompts.prompt_builder import BuiltPrompt
        from app.utils.tokenizer import count_tokens
        return BuiltPrompt(
            system_message=system,
            user_message=f"{context}\n\nQUESTION: {question}",
            context_chunks_used=0,
            total_context_tokens=count_tokens(context),
        )

    def _format_chunks_as_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a readable context block."""
        if not chunks:
            return NO_EVIDENCE_MSG
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[{i}] {c.title} (page {c.page}, section: {c.section or 'N/A'})\n"
                f"Similarity: {c.similarity_score:.3f}\n"
                f"{c.chunk_text}"
            )
        return "\n---\n".join(parts)

    def _error_result(self, error: Exception | str, elapsed_ms: float = 0.0) -> AgentResult:
        """Return a failure AgentResult with error details."""
        msg = str(error)
        logger.error("%s failed: %s", self.agent_name, msg)
        return AgentResult(
            agent_name=self.agent_name,
            success=False,
            confidence=Confidence.NONE,
            summary=f"Agent failed: {msg}",
            error=msg,
            processing_time_ms=elapsed_ms,
        )
