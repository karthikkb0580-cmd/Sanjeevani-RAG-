"""
app/orchestrator/orchestrator.py

Central Multi-Agent Orchestrator for Sanjeevani AI.

Execution flow:
  1. Receive AnalysisRequest
  2. Instantiate all parallel agents (1-8)
  3. Run them concurrently with asyncio.gather()
  4. Merge results into AgentResults dict
  5. Run Agent 9 (ResearchSummarizationAgent) on merged results
  6. Run Agent 10 (FinalReasoningAgent) on all results + summarization
  7. Assemble FinalAnalysisReport
  8. Return structured report (or stream)

Design principles:
  - Open/Closed: Add new agents by registering in agents/__init__.py
  - Error isolation: One failed agent never aborts the pipeline
  - Dependency Injection: All agents share the same Retriever + LLM client
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import AsyncGenerator

from app.agents import (
    PARALLEL_AGENTS,
    FinalReasoningAgent,
    ResearchSummarizationAgent,
)
from app.agents.base_agent import BaseAgent
from app.embeddings.embedding_service import get_embedding_provider
from app.llm.openai_client import get_llm_client
from app.retrieval.retriever import Retriever
from app.schemas.agents import (
    AgentResult,
    ChemicalAvailabilityResult,
    Confidence,
    DrugLikenessResult,
    LiteratureResult,
    NoveltyResult,
    PatentResult,
    ResearchSummaryResult,
    SafetyResult,
    SyntheticRouteResult,
    ToxicityResult,
)
from app.schemas.analysis import (
    AnalysisRequest,
    ConfidenceAssessment,
    FinalAnalysisReport,
)

logger = logging.getLogger(__name__)


class AnalysisOrchestrator:
    """
    Central orchestrator for the Sanjeevani multi-agent RAG pipeline.

    Instantiate once at application startup (or per-request for isolation).
    Uses a shared Retriever and LLM client across all agents for efficiency.
    """

    def __init__(self) -> None:
        # Shared infrastructure
        self._retriever  = Retriever()
        self._llm        = get_llm_client()
        self._embedder   = get_embedding_provider()

        # Instantiate all registered parallel agents
        self._parallel_agents: list[BaseAgent] = [
            AgentClass(retriever=self._retriever, llm_client=self._llm)
            for AgentClass in PARALLEL_AGENTS
        ]
        # Sequential agents
        self._summarization_agent = ResearchSummarizationAgent(
            retriever=self._retriever, llm_client=self._llm
        )
        self._final_agent = FinalReasoningAgent(
            retriever=self._retriever, llm_client=self._llm
        )

        logger.info(
            "Orchestrator initialized with %d parallel agents",
            len(self._parallel_agents),
        )

    async def analyze(self, request: AnalysisRequest) -> FinalAnalysisReport:
        """
        Run the full multi-agent analysis pipeline.

        Args:
            request: AnalysisRequest with query/molecule and retrieval params

        Returns:
            FinalAnalysisReport with all 13 output sections
        """
        request_id = str(uuid.uuid4())
        start      = time.perf_counter()

        logger.info(
            "=== ORCHESTRATOR START | request_id=%s | query='%s…' ===",
            request_id,
            request.effective_query()[:80],
        )

        # ── Determine which agents to run ────────────────────────────────────
        active_agents = self._filter_agents(request)

        # ── Phase 1: Parallel agent execution (Agents 1-8) ───────────────────
        logger.info("Phase 1: Running %d agents in parallel …", len(active_agents))
        phase1_start = time.perf_counter()

        parallel_results_list: list[AgentResult] = await asyncio.gather(
            *[agent.run(request) for agent in active_agents],
            return_exceptions=False,   # Agents catch their own exceptions
        )

        phase1_elapsed = (time.perf_counter() - phase1_start) * 1000
        logger.info("Phase 1 complete in %.0f ms", phase1_elapsed)

        # Map agent name → AgentResult
        agent_results: dict[str, AgentResult] = {
            result.agent_name: result
            for result in parallel_results_list
        }

        succeeded = sum(1 for r in agent_results.values() if r.success)
        failed    = len(agent_results) - succeeded
        logger.info("Phase 1 results: %d succeeded, %d failed", succeeded, failed)

        # ── Phase 2: Research Summarization (Agent 9) ─────────────────────────
        logger.info("Phase 2: Running ResearchSummarizationAgent …")
        phase2_start = time.perf_counter()

        summarization_result = await self._summarization_agent.run(
            request, agent_results=agent_results
        )
        phase2_elapsed = (time.perf_counter() - phase2_start) * 1000
        logger.info(
            "Phase 2 complete in %.0f ms | confidence=%s",
            phase2_elapsed,
            summarization_result.confidence.value,
        )

        # ── Phase 3: Final Reasoning (Agent 10) ───────────────────────────────
        logger.info("Phase 3: Running FinalReasoningAgent …")
        phase3_start = time.perf_counter()

        final_result = await self._final_agent.run(
            request,
            agent_results=agent_results,
            summarization_result=summarization_result,
        )
        phase3_elapsed = (time.perf_counter() - phase3_start) * 1000
        logger.info("Phase 3 complete in %.0f ms", phase3_elapsed)

        # ── Phase 4: Assemble FinalAnalysisReport ────────────────────────────
        total_elapsed_ms = (time.perf_counter() - start) * 1000
        report = self._assemble_report(
            request_id=request_id,
            request=request,
            agent_results=agent_results,
            summarization_result=summarization_result,
            final_result=final_result,
            total_ms=total_elapsed_ms,
        )

        logger.info(
            "=== ORCHESTRATOR COMPLETE | request_id=%s | %.0f ms | confidence=%s ===",
            request_id,
            total_elapsed_ms,
            report.confidence_assessment.overall.value,
        )

        return report

    async def stream_analyze(
        self,
        request: AnalysisRequest,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the analysis as Server-Sent Events (SSE).

        Yields progress events during parallel execution,
        then streams the final report.
        """
        request_id = str(uuid.uuid4())
        start      = time.perf_counter()

        yield self._sse_event("start", {
            "request_id": request_id,
            "message": "Analysis started",
            "total_agents": len(PARALLEL_AGENTS) + 2,
        })

        active_agents = self._filter_agents(request)

        # Phase 1: Parallel with individual progress events
        yield self._sse_event("phase", {"phase": 1, "message": "Running parallel agents …"})

        tasks = [agent.run(request) for agent in active_agents]
        agent_results: dict[str, AgentResult] = {}

        for coro in asyncio.as_completed(tasks):
            result: AgentResult = await coro
            agent_results[result.agent_name] = result
            yield self._sse_event("agent_complete", {
                "agent": result.agent_name,
                "success": result.success,
                "confidence": result.confidence.value,
                "chunks_retrieved": result.chunks_retrieved,
                "processing_time_ms": result.processing_time_ms,
            })

        # Phase 2
        yield self._sse_event("phase", {"phase": 2, "message": "Synthesizing research …"})
        summarization_result = await self._summarization_agent.run(
            request, agent_results=agent_results
        )
        yield self._sse_event("agent_complete", {
            "agent": summarization_result.agent_name,
            "success": summarization_result.success,
        })

        # Phase 3
        yield self._sse_event("phase", {"phase": 3, "message": "Generating final report …"})
        final_result = await self._final_agent.run(
            request,
            agent_results=agent_results,
            summarization_result=summarization_result,
        )

        total_elapsed_ms = (time.perf_counter() - start) * 1000

        report = self._assemble_report(
            request_id=request_id,
            request=request,
            agent_results=agent_results,
            summarization_result=summarization_result,
            final_result=final_result,
            total_ms=total_elapsed_ms,
        )

        yield self._sse_event("complete", {"request_id": request_id})
        yield self._sse_event("report", report.model_dump(mode="json"))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_agents(self, request: AnalysisRequest) -> list[BaseAgent]:
        """Filter agents based on request.agents_enabled (if provided)."""
        if not request.agents_enabled:
            return self._parallel_agents
        enabled = set(request.agents_enabled)
        return [a for a in self._parallel_agents if a.agent_name in enabled]

    def _assemble_report(
        self,
        request_id: str,
        request: AnalysisRequest,
        agent_results: dict[str, AgentResult],
        summarization_result: AgentResult,
        final_result: AgentResult,
        total_ms: float,
    ) -> FinalAnalysisReport:
        """Assemble all agent results into the final structured report."""

        # Extract per-agent structured details
        lit     = self._extract_details(agent_results, "LiteratureRetrievalAgent",   LiteratureResult)
        synth   = self._extract_details(agent_results, "SyntheticRouteAgent",         SyntheticRouteResult)
        avail   = self._extract_details(agent_results, "ChemicalAvailabilityAgent",   ChemicalAvailabilityResult)
        patent  = self._extract_details(agent_results, "PatentRetrievalAgent",        PatentResult)
        tox     = self._extract_details(agent_results, "ToxicityAgent",               ToxicityResult)
        safety  = self._extract_details(agent_results, "SafetyAgent",                 SafetyResult)
        dl      = self._extract_details(agent_results, "DrugLikenessAgent",           DrugLikenessResult)
        novelty = self._extract_details(agent_results, "NoveltyAgent",                NoveltyResult)

        # Research gaps from summarization
        research_gaps: list[str] = []
        if summarization_result.success and summarization_result.details:
            research_gaps = summarization_result.details.get("research_gaps", [])

        # All unique references
        references = self._build_references(agent_results)

        # Confidence assessment
        conf = ConfidenceAssessment()
        conf.overall_from_agents(agent_results)

        # Summary details
        sum_details = summarization_result.details if summarization_result.success else {}

        succeeded = sum(1 for r in agent_results.values() if r.success)
        failed    = len(agent_results) - succeeded

        return FinalAnalysisReport(
            request_id=request_id,
            query=request.effective_query(),
            molecule_input=request.molecule,

            # Section 1
            executive_summary=self._extract_section(
                final_result.summary, "EXECUTIVE SUMMARY"
            ) or final_result.summary[:800],

            # Section 2
            molecule_overview=self._extract_section(
                final_result.summary, "MOLECULE OVERVIEW"
            ),

            # Sections 3-10 (structured data from agents)
            literature_findings=lit,
            synthetic_route_summary=synth,
            reagent_availability_status=avail,
            patent_landscape=patent,
            toxicity_evidence_summary=tox,
            laboratory_safety_summary=safety,
            drug_likeness_assessment=dl,
            novelty_assessment=novelty,

            # Section 11
            research_gaps=research_gaps,

            # Section 12
            confidence_assessment=conf,

            # Section 13
            references=references,

            # Metadata
            research_summary_narrative=sum_details.get(
                "synthesis_narrative", summarization_result.summary[:1000]
            ),
            final_reasoning_narrative=final_result.summary,
            agent_results=agent_results,
            agents_succeeded=succeeded,
            agents_failed=failed,
            total_processing_time_ms=round(total_ms, 2),
            llm_model=self._llm.model_name,
            embedding_model=self._embedder.model_name,
        )

    @staticmethod
    def _extract_details(
        agent_results: dict[str, AgentResult],
        agent_name: str,
        model_class: type,
    ):
        """Safely extract and validate agent details into a Pydantic model."""
        result = agent_results.get(agent_name)
        if not result or not result.success or not result.details:
            return None
        try:
            return model_class.model_validate(result.details)
        except Exception as exc:
            logger.warning("Failed to parse %s details: %s", agent_name, exc)
            return None

    @staticmethod
    def _build_references(agent_results: dict[str, AgentResult]) -> list[str]:
        """Build deduplicated reference list from all agent evidence."""
        seen: set[str] = set()
        refs: list[str] = []
        for result in agent_results.values():
            for ev in result.evidence:
                ref = f"{ev.title} (p. {ev.page}, §{ev.section or 'N/A'})"
                if ref not in seen and ev.title:
                    seen.add(ref)
                    refs.append(ref)
        return sorted(refs)

    @staticmethod
    def _extract_section(text: str, header: str) -> str:
        """Extract a named section from the final reasoning narrative."""
        if not text:
            return ""
        import re
        m = re.search(
            rf"{re.escape(header)}:\s*\n(.+?)(?:\n[A-Z\s]+:|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        return m.group(1).strip() if m else ""

    @staticmethod
    def _sse_event(event: str, data: dict) -> str:
        """Format a Server-Sent Event string."""
        import json
        return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_orchestrator: AnalysisOrchestrator | None = None


def get_orchestrator() -> AnalysisOrchestrator:
    """Return the module-level orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AnalysisOrchestrator()
    return _orchestrator
