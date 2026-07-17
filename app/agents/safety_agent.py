"""
app/agents/safety_agent.py

Agent 6 – Safety Information Agent

Responsibilities:
- Retrieve laboratory handling information from indexed references
- Return: handling precautions, storage recommendations, general lab considerations
- Never generate hazardous procedural guidance
"""

from __future__ import annotations

import logging
import time

from app.agents.base_agent import NO_EVIDENCE_MSG, BaseAgent
from app.schemas.agents import AgentResult, Confidence, SafetyResult
from app.schemas.analysis import AnalysisRequest

logger = logging.getLogger(__name__)


SAFETY_SYSTEM_PROMPT = """\
You are a laboratory safety information analyst.
You review indexed scientific references, SDS-like data, and safety literature.

Your task:
1. Extract laboratory handling precautions from the retrieved context.
2. Identify storage recommendations.
3. List PPE requirements if mentioned.
4. Note disposal considerations.

Critical rules:
- Do NOT generate hazardous synthesis or procedural instructions.
- Do NOT fabricate safety data not present in the retrieved context.
- If no safety data is found, state:
  "No laboratory safety data was found in the indexed knowledge base."
- Always cite source documents.
- Always include the institutional EHS disclaimer.

Output sections:
HANDLING PRECAUTIONS: [list]
STORAGE: [list]
PPE: [list]
DISPOSAL: [list]
REFERENCES: [list]
"""


class SafetyAgent(BaseAgent):
    """
    Agent 6: Safety Information Agent

    Retrieves lab safety data from indexed sources, never generates
    hazardous procedural guidance.
    """

    @property
    def agent_name(self) -> str:
        return "SafetyAgent"

    async def run(self, request: AnalysisRequest) -> AgentResult:
        start = time.perf_counter()
        try:
            primary_query = request.effective_query()

            probe_queries = [
                f"{primary_query} laboratory safety handling precautions",
                f"{primary_query} storage conditions temperature stability",
                f"{primary_query} PPE personal protective equipment gloves",
                f"{primary_query} disposal waste management environmental",
                f"{primary_query} SDS safety data sheet MSDS hazard",
            ]

            chunks     = await self._retrieve_multi(probe_queries, request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            confidence  = self._derive_confidence(chunks)
            evidence   = self._chunks_to_evidence(chunks)

            if chunks:
                context  = self._format_chunks_as_context(chunks[:10])
                question = (
                    f"From the retrieved passages, extract laboratory safety information for: '{primary_query}'. "
                    f"List handling precautions, storage conditions, PPE, and disposal notes. "
                    f"Never generate hazardous instructions. Cite sources."
                )
                prompt   = self._build_simple_prompt(SAFETY_SYSTEM_PROMPT, context, question)
                llm_resp = await self._llm.complete(prompt)
                summary  = llm_resp.content

                safety_result = self._parse_safety_response(summary, evidence)
            else:
                summary = (
                    "No laboratory safety data was found in the indexed knowledge base. "
                    "Please consult the manufacturer's Safety Data Sheet (SDS/MSDS) and "
                    "your institutional Environmental Health and Safety (EHS) office."
                )
                safety_result = SafetyResult(
                    handling_precautions=["Consult institutional SDS/MSDS"],
                    storage_recommendations=["Consult manufacturer guidelines"],
                )

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                confidence=confidence,
                summary=summary,
                evidence=evidence,
                details=safety_result.model_dump(),
                processing_time_ms=elapsed_ms,
                chunks_retrieved=len(chunks),
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._error_result(exc, elapsed_ms)

    @staticmethod
    def _parse_safety_response(text: str, evidence: list) -> SafetyResult:
        """Parse structured safety fields from LLM narrative."""
        import re

        def _extract_list(header: str) -> list[str]:
            m = re.search(
                rf"{header}:\s*(.+?)(?:\n[A-Z]|$)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if not m:
                return []
            raw = m.group(1).strip()
            # Split on bullet points, commas, or semicolons
            items = re.split(r"[;,\n•\-]", raw)
            return [i.strip() for i in items if len(i.strip()) > 3]

        handling    = _extract_list("HANDLING PRECAUTIONS")
        storage     = _extract_list("STORAGE")
        ppe         = _extract_list("PPE")
        disposal    = _extract_list("DISPOSAL")
        references  = [f"{e.title}, p.{e.page}" for e in evidence[:5] if e.title]

        # Defaults if LLM didn't use structured format
        if not handling:
            handling = ["Handle with care per laboratory safety guidelines"]
        if not storage:
            storage = ["Store per manufacturer SDS recommendations"]

        return SafetyResult(
            handling_precautions=handling,
            storage_recommendations=storage,
            ppe_recommendations=ppe,
            disposal_notes=disposal,
            supporting_references=references,
        )
