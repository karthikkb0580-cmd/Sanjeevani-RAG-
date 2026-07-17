"""
app/agents/synthetic_route_agent.py

Agent 2 – Synthetic Route Information Agent

Responsibilities:
- Retrieve published synthesis routes from the knowledge base
- Compare alternative synthetic pathways
- Estimate route complexity using retrieved literature
- Identify starting materials
- Return: reaction sequence, reagents, steps, complexity, citations
- Clearly distinguish literature-derived vs inferred information
"""

from __future__ import annotations

import logging
import re
import time

from app.agents.base_agent import NO_EVIDENCE_MSG, BaseAgent
from app.schemas.agents import (
    AgentResult,
    Confidence,
    SyntheticRouteResult,
    SyntheticStep,
)
from app.schemas.analysis import AnalysisRequest

logger = logging.getLogger(__name__)


SYNTHETIC_ROUTE_SYSTEM_PROMPT = """\
You are an expert organic chemistry synthesis analyst.
You receive retrieved passages from indexed chemistry literature.

Your task:
1. Identify all reported synthesis routes for the target compound.
2. List each synthetic step with: reagents, conditions, and citations.
3. Estimate the total number of steps.
4. Assess route complexity: Simple | Moderate | Complex | Highly Complex
5. Identify key starting materials.
6. Note whether each piece of information is LITERATURE-DERIVED or INFERRED.

Critical rules:
- Do NOT invent reaction steps, reagents, or conditions.
- If no synthesis route is found, say: "No synthesis route was found in the indexed knowledge base."
- Never hallucinate chemical reagents or conditions.
- Always cite the source document for each claim.
- Separate LITERATURE-DERIVED facts from INFERRED estimates.

Response format:
ROUTES FOUND: [number]
COMPLEXITY: [Simple|Moderate|Complex|Highly Complex]
STEPS: [number or Unknown]
STARTING MATERIALS: [list]
REAGENTS: [list]
SUMMARY: [narrative description]
"""


class SyntheticRouteAgent(BaseAgent):
    """
    Agent 2: Synthetic Route Information Agent

    Retrieves synthesis routes from indexed chemistry literature,
    never invents reagents or conditions.
    """

    @property
    def agent_name(self) -> str:
        return "SyntheticRouteAgent"

    async def run(self, request: AnalysisRequest) -> AgentResult:
        start = time.perf_counter()
        try:
            primary_query = request.effective_query()

            # Synthesis-specific probe queries
            probe_queries = [
                f"{primary_query} synthesis route preparation method",
                f"{primary_query} synthetic pathway starting materials reagents",
                f"{primary_query} total synthesis steps procedure",
                f"{primary_query} chemical synthesis retrosynthesis",
            ]

            chunks = await self._retrieve_multi(probe_queries, request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            confidence  = self._derive_confidence(chunks)
            evidence    = self._chunks_to_evidence(chunks)

            if chunks:
                context  = self._format_chunks_as_context(chunks[:12])
                question = (
                    f"From the retrieved passages, extract all synthesis routes for: '{primary_query}'. "
                    f"List synthetic steps with reagents and conditions. "
                    f"Estimate complexity and number of steps. "
                    f"Explicitly state: LITERATURE-DERIVED or INFERRED for every claim."
                )
                prompt    = self._build_simple_prompt(SYNTHETIC_ROUTE_SYSTEM_PROMPT, context, question)
                llm_resp  = await self._llm.complete(prompt)
                summary   = llm_resp.content

                # Parse structured fields from LLM response
                route_result = self._parse_route_response(summary, evidence)
            else:
                summary      = NO_EVIDENCE_MSG
                route_result = SyntheticRouteResult(
                    routes_found=0,
                    is_literature_derived=False,
                )

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                confidence=confidence,
                summary=summary,
                evidence=evidence,
                details=route_result.model_dump(),
                processing_time_ms=elapsed_ms,
                chunks_retrieved=len(chunks),
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._error_result(exc, elapsed_ms)

    @staticmethod
    def _parse_route_response(text: str, evidence: list) -> SyntheticRouteResult:
        """Extract structured fields from LLM narrative response."""

        def _extract(pattern: str, default: str = "") -> str:
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else default

        routes_found_str   = _extract(r"ROUTES FOUND:\s*(\d+)", "0")
        complexity_str     = _extract(r"COMPLEXITY:\s*(\w[\w\s]*)", "Unknown")
        steps_str          = _extract(r"STEPS:\s*(\w[\w\s]*)", "Unknown")
        starting_mat_str   = _extract(r"STARTING MATERIALS:\s*(.+?)(?:\n[A-Z]|$)")
        reagents_str       = _extract(r"REAGENTS:\s*(.+?)(?:\n[A-Z]|$)")
        summary_str        = _extract(r"SUMMARY:\s*(.+?)$")

        routes_found = int(routes_found_str) if routes_found_str.isdigit() else 0
        steps: int | None = None
        if steps_str.isdigit():
            steps = int(steps_str)

        # Extract supporting citations from evidence
        supporting_citations = [
            f"{e.title}, p.{e.page}"
            for e in evidence[:8]
            if e.title
        ]

        return SyntheticRouteResult(
            routes_found=routes_found,
            reaction_sequence=[],   # Populated by LLM narrative
            reported_reagents=[r.strip() for r in reagents_str.split(",") if r.strip()],
            number_of_steps=steps,
            estimated_complexity=complexity_str.strip()[:50],
            starting_materials=[s.strip() for s in starting_mat_str.split(",") if s.strip()],
            supporting_citations=supporting_citations,
            is_literature_derived=routes_found > 0,
        )
