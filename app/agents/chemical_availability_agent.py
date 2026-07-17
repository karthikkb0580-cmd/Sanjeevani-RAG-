"""
app/agents/chemical_availability_agent.py

Agent 3 – Chemical Availability Agent

Responsibilities:
- Retrieve availability information from indexed supplier datasets
- Never invent availability
- If no data source exists, report unverified status
- Return: availability status, possible suppliers, evidence source
"""

from __future__ import annotations

import logging
import time

from app.agents.base_agent import NO_EVIDENCE_MSG, BaseAgent
from app.schemas.agents import (
    AgentResult,
    AvailabilityStatus,
    ChemicalAvailabilityResult,
    Confidence,
)
from app.schemas.analysis import AnalysisRequest

logger = logging.getLogger(__name__)


AVAILABILITY_SYSTEM_PROMPT = """\
You are a chemical supply chain analyst reviewing indexed literature and supplier data.

Your task:
1. Determine the availability of the target compound or its precursors.
2. Identify any mentioned suppliers, catalog references, or commercial sources.
3. State the evidence source explicitly.

Critical rules:
- Do NOT invent supplier names, catalog numbers, or prices.
- If no availability information is found in the indexed data, clearly state:
  "Availability could not be verified from the indexed knowledge base."
- Distinguish between: COMMERCIALLY AVAILABLE | LIMITED AVAILABILITY | NOT AVAILABLE | UNVERIFIED
- Cite the source document for any availability claim.
"""


class ChemicalAvailabilityAgent(BaseAgent):
    """
    Agent 3: Chemical Availability Agent

    Retrieves availability from the indexed knowledge base only.
    Never fabricates supplier information.
    """

    @property
    def agent_name(self) -> str:
        return "ChemicalAvailabilityAgent"

    async def run(self, request: AnalysisRequest) -> AgentResult:
        start = time.perf_counter()
        try:
            primary_query = request.effective_query()

            probe_queries = [
                f"{primary_query} commercial availability supplier catalog",
                f"{primary_query} chemical purchase procurement sigma aldrich",
                f"{primary_query} reagent stock laboratory supply",
            ]

            chunks    = await self._retrieve_multi(probe_queries, request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            evidence  = self._chunks_to_evidence(chunks)

            if chunks:
                confidence = self._derive_confidence(chunks)
                context    = self._format_chunks_as_context(chunks[:8])
                question   = (
                    f"Based on the retrieved passages, assess the commercial availability "
                    f"of: '{primary_query}'. "
                    f"Identify any suppliers, catalog references, or availability notes. "
                    f"State your confidence and cite sources."
                )
                prompt   = self._build_simple_prompt(AVAILABILITY_SYSTEM_PROMPT, context, question)
                llm_resp = await self._llm.complete(prompt)
                summary  = llm_resp.content

                avail_result = self._parse_availability(llm_resp.content, evidence)
            else:
                confidence = Confidence.NONE
                summary    = (
                    "Availability could not be verified from the indexed knowledge base. "
                    "No supplier or catalog data was found in the indexed documents. "
                    "To verify availability, consult commercial databases such as "
                    "Sigma-Aldrich, TCI, Alfa Aesar, or Merck directly."
                )
                avail_result = ChemicalAvailabilityResult(
                    availability_status=AvailabilityStatus.UNVERIFIED,
                    evidence_source="No supporting evidence found in indexed knowledge base.",
                    notes=summary,
                )

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                confidence=confidence,
                summary=summary,
                evidence=evidence,
                details=avail_result.model_dump(),
                processing_time_ms=elapsed_ms,
                chunks_retrieved=len(chunks),
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._error_result(exc, elapsed_ms)

    @staticmethod
    def _parse_availability(
        text: str,
        evidence: list,
    ) -> ChemicalAvailabilityResult:
        """Parse availability status from LLM response."""
        import re

        text_lower = text.lower()

        if "commercially available" in text_lower or "available from" in text_lower:
            status = AvailabilityStatus.AVAILABLE
        elif "limited" in text_lower:
            status = AvailabilityStatus.LIMITED
        elif "not available" in text_lower or "unavailable" in text_lower:
            status = AvailabilityStatus.NOT_AVAILABLE
        else:
            status = AvailabilityStatus.UNVERIFIED

        # Try to extract supplier mentions (simple heuristic)
        supplier_patterns = [
            r"sigma.?aldrich", r"sigma-aldrich", r"merck", r"tci", r"alfa aesar",
            r"cayman chemical", r"santa cruz", r"abcam", r"combi.?blocks",
            r"enamine", r"fluorochem", r"oakwood chemical",
        ]
        suppliers = []
        for pat in supplier_patterns:
            if re.search(pat, text_lower):
                suppliers.append(pat.replace(r".", "").replace("?", "").title())

        sources = list({e.title for e in evidence[:5] if e.title})
        evidence_source = ", ".join(sources) if sources else "Indexed knowledge base"

        return ChemicalAvailabilityResult(
            availability_status=status,
            possible_suppliers=suppliers,
            evidence_source=evidence_source,
            notes="Availability assessed from indexed scientific literature only.",
        )
