"""
app/agents/summarization_agent.py

Agent 9 – Research Summarization Agent

Responsibilities:
- Receive outputs from ALL previous agents (1-8)
- Summarize all retrieved papers
- Extract: key findings, limitations, consensus, research gaps
- Returns: ResearchSummaryResult
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.agents.base_agent import BaseAgent
from app.schemas.agents import AgentResult, Confidence, ResearchSummaryResult
from app.schemas.analysis import AnalysisRequest

logger = logging.getLogger(__name__)


SUMMARIZATION_SYSTEM_PROMPT = """\
You are a senior scientific research synthesizer.
You receive the structured outputs of multiple specialized analysis agents
and must synthesize them into a coherent research summary.

Your task:
1. Identify the KEY FINDINGS across all agent outputs.
2. Note any LIMITATIONS or gaps in the evidence.
3. Identify CONSENSUS points (where multiple agents agree).
4. Identify RESEARCH GAPS (what remains unknown or unstudied).
5. Note any CONFLICTING FINDINGS and explicitly state the disagreement.

Output format:
KEY FINDINGS:
[bullet list]

LIMITATIONS:
[bullet list]

CONSENSUS:
[bullet list]

RESEARCH GAPS:
[bullet list]

SYNTHESIS NARRATIVE:
[1-2 paragraph synthesis]

Rules:
- Never discard evidence from any agent.
- Never fabricate findings.
- Conflicting data must be explicitly reported.
- Base your synthesis ONLY on the agent outputs provided.
"""


class ResearchSummarizationAgent(BaseAgent):
    """
    Agent 9: Research Summarization Agent

    Receives a dict of all prior agent results and synthesizes
    a comprehensive research summary.
    """

    @property
    def agent_name(self) -> str:
        return "ResearchSummarizationAgent"

    async def run(
        self,
        request: AnalysisRequest,
        agent_results: dict[str, AgentResult] | None = None,
    ) -> AgentResult:
        start = time.perf_counter()
        try:
            if not agent_results:
                return self._error_result("No agent results provided to summarize", 0.0)

            # Build a comprehensive summary of all prior agent outputs
            agent_summary_text = self._format_agent_results(agent_results)

            primary_query = request.effective_query()
            question = (
                f"Synthesize the following multi-agent analysis results for: '{primary_query}'. "
                f"Extract key findings, limitations, consensus points, and research gaps. "
                f"Report any conflicting findings explicitly."
            )

            prompt   = self._build_simple_prompt(
                SUMMARIZATION_SYSTEM_PROMPT,
                agent_summary_text,
                question,
            )
            llm_resp = await self._llm.complete(prompt)
            summary  = llm_resp.content

            # Parse structured sections from LLM response
            parsed = self._parse_summary_sections(summary)

            # Count total papers across all agents
            total_papers = self._count_total_papers(agent_results)

            summary_result = ResearchSummaryResult(
                key_findings=parsed.get("key_findings", []),
                limitations=parsed.get("limitations", []),
                consensus_points=parsed.get("consensus", []),
                research_gaps=parsed.get("research_gaps", []),
                paper_count=total_papers,
                synthesis_narrative=parsed.get("narrative", summary[:1000]),
            )

            # Compute confidence based on prior agent success rate
            success_count = sum(1 for r in agent_results.values() if r.success)
            total_count   = len(agent_results)
            if success_count == total_count and total_count >= 6:
                confidence = Confidence.HIGH
            elif success_count >= total_count // 2:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            elapsed_ms = (time.perf_counter() - start) * 1000

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                confidence=confidence,
                summary=summary,
                evidence=[],    # Summarization agent synthesizes, doesn't retrieve
                details=summary_result.model_dump(),
                processing_time_ms=elapsed_ms,
                chunks_retrieved=0,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._error_result(exc, elapsed_ms)

    @staticmethod
    def _format_agent_results(agent_results: dict[str, AgentResult]) -> str:
        """Format all agent results into a single context block for the LLM."""
        parts: list[str] = []
        for name, result in agent_results.items():
            status  = "✓ SUCCESS" if result.success else "✗ FAILED"
            conf    = result.confidence.value.upper()
            summary = result.summary[:800] if result.summary else "No summary available."
            parts.append(
                f"=== {name} [{status}] [Confidence: {conf}] ===\n"
                f"{summary}\n"
                f"Chunks retrieved: {result.chunks_retrieved}\n"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _parse_summary_sections(text: str) -> dict[str, Any]:
        """Parse structured sections from the LLM summarization response."""
        import re

        def _extract_section(header: str) -> list[str]:
            m = re.search(
                rf"{header}:\s*\n(.*?)(?:\n[A-Z\s]+:|\Z)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if not m:
                return []
            raw = m.group(1).strip()
            items = re.split(r"\n[-•*]\s*|\n\d+\.\s*", raw)
            return [i.strip() for i in items if len(i.strip()) > 5]

        def _extract_narrative() -> str:
            m = re.search(
                r"SYNTHESIS NARRATIVE:\s*\n(.+?)(?:\n[A-Z\s]+:|$)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            return m.group(1).strip() if m else ""

        return {
            "key_findings":  _extract_section("KEY FINDINGS"),
            "limitations":   _extract_section("LIMITATIONS"),
            "consensus":     _extract_section("CONSENSUS"),
            "research_gaps": _extract_section("RESEARCH GAPS"),
            "narrative":     _extract_narrative(),
        }

    @staticmethod
    def _count_total_papers(agent_results: dict[str, AgentResult]) -> int:
        """Count total unique papers across all agent evidence."""
        seen: set[str] = set()
        for result in agent_results.values():
            for ev in result.evidence:
                if ev.document_id:
                    seen.add(ev.document_id)
        return len(seen)
