"""
app/agents/toxicity_agent.py

Agent 5 – Toxicity Information Agent

Responsibilities:
- Retrieve toxicity information from trusted indexed scientific sources
- Separate observed evidence from predictions
- Report: known hazards, exposure considerations, published toxicity classifications
- Never provide medical or regulatory conclusions
"""

from __future__ import annotations

import logging
import time

from app.agents.base_agent import NO_EVIDENCE_MSG, BaseAgent
from app.schemas.agents import (
    AgentResult,
    Confidence,
    ToxicityClassification,
    ToxicityResult,
)
from app.schemas.analysis import AnalysisRequest

logger = logging.getLogger(__name__)


TOXICITY_SYSTEM_PROMPT = """\
You are a toxicology information analyst reviewing indexed scientific literature.

Your task:
1. Extract all reported toxicity information from the retrieved passages.
2. List known hazards (LD50, LC50, NOAEL, LOAEL if mentioned).
3. Identify GHS/UN hazard classifications if reported.
4. Note exposure routes and considerations.
5. Distinguish: OBSERVED (published study data) vs PREDICTED (computational model).

Critical rules:
- Do NOT fabricate toxicity values, LD50 values, or safety classifications.
- Do NOT make medical or regulatory conclusions.
- If no toxicity data is found in the context, state:
  "No toxicity data was found in the indexed knowledge base for this compound."
- Always cite the source document for each toxicity claim.
- Clearly flag: [OBSERVED FROM LITERATURE] or [COMPUTATIONAL PREDICTION]

Disclaimer to include:
"This information is retrieved from scientific literature for research purposes only.
 It does not constitute medical advice, regulatory guidance, or safety certification."
"""


class ToxicityAgent(BaseAgent):
    """
    Agent 5: Toxicity Information Agent

    Retrieves evidence-backed toxicity data, never fabricates values.
    """

    @property
    def agent_name(self) -> str:
        return "ToxicityAgent"

    async def run(self, request: AnalysisRequest) -> AgentResult:
        start = time.perf_counter()
        try:
            primary_query = request.effective_query()

            probe_queries = [
                f"{primary_query} toxicity LD50 LC50 hazard",
                f"{primary_query} toxic effects carcinogenicity mutagenicity",
                f"{primary_query} GHS classification hazard statement",
                f"{primary_query} acute toxicity chronic exposure NOAEL LOAEL",
                f"{primary_query} adverse effects safety pharmacology",
            ]

            chunks     = await self._retrieve_multi(probe_queries, request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            confidence  = self._derive_confidence(chunks)
            evidence   = self._chunks_to_evidence(chunks)

            if chunks:
                context  = self._format_chunks_as_context(chunks[:12])
                question = (
                    f"Extract all toxicity information from the retrieved passages for: '{primary_query}'. "
                    f"List hazards, classifications (GHS if available), exposure routes. "
                    f"Flag OBSERVED vs PREDICTED. Never fabricate values. Cite sources."
                )
                prompt   = self._build_simple_prompt(TOXICITY_SYSTEM_PROMPT, context, question)
                llm_resp = await self._llm.complete(prompt)
                summary  = llm_resp.content

                tox_result = self._build_toxicity_result(summary, evidence)
            else:
                summary    = NO_EVIDENCE_MSG
                tox_result = ToxicityResult(
                    data_gap_note=(
                        "No toxicity data was found in the indexed knowledge base. "
                        "Consult official databases (PubChem, ChemSpider, ECHA, EPA) for "
                        "authoritative toxicity information."
                    )
                )

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                confidence=confidence,
                summary=summary,
                evidence=evidence,
                details=tox_result.model_dump(),
                processing_time_ms=elapsed_ms,
                chunks_retrieved=len(chunks),
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._error_result(exc, elapsed_ms)

    @staticmethod
    def _build_toxicity_result(summary: str, evidence: list) -> ToxicityResult:
        """Parse toxicity result from LLM summary."""
        import re

        text_lower = summary.lower()

        # Extract hazard keywords
        hazard_keywords = [
            "flammable", "toxic", "corrosive", "irritant", "carcinogen",
            "mutagen", "teratogen", "sensitizer", "oxidizer", "explosive",
            "ld50", "lc50", "noael", "loael", "reproductive toxicity",
        ]
        known_hazards = [
            kw.upper()
            for kw in hazard_keywords
            if kw in text_lower
        ]

        # Extract exposure considerations
        exposure_keywords = ["inhalation", "dermal", "oral", "eye contact", "ingestion"]
        exposures = [e.capitalize() for e in exposure_keywords if e in text_lower]

        # Build citations from evidence
        citations = [
            f"{e.title}, p.{e.page}"
            for e in evidence[:8]
            if e.title
        ]

        # Extract GHS classification if present
        classifications: list[ToxicityClassification] = []
        ghs_pattern = re.compile(r"GHS\s*\d+|Acute\s+Tox(?:icity)?\s+\d+", re.IGNORECASE)
        for m in ghs_pattern.finditer(summary):
            classifications.append(
                ToxicityClassification(
                    classification=m.group(0),
                    source="Retrieved literature",
                    is_predicted=False,
                )
            )

        return ToxicityResult(
            known_hazards=known_hazards,
            exposure_considerations=exposures,
            classifications=classifications,
            relevant_citations=citations,
            data_gap_note="" if evidence else "Limited toxicity data available in indexed knowledge base.",
        )
