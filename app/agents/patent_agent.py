"""
app/agents/patent_agent.py

Agent 4 – Patent Retrieval Agent

Responsibilities:
- Retrieve patent information from the indexed knowledge base
- Prioritize Indian patents (IN), also WIPO, USPTO, EPO
- Return: patent numbers, dates, legal status, relevant claims, similarity
- Never fabricate patent numbers or claims
"""

from __future__ import annotations

import logging
import re
import time

from app.agents.base_agent import NO_EVIDENCE_MSG, BaseAgent
from app.schemas.agents import (
    AgentResult,
    Confidence,
    PatentRecord,
    PatentResult,
)
from app.schemas.analysis import AnalysisRequest

logger = logging.getLogger(__name__)


PATENT_SYSTEM_PROMPT = """\
You are a patent analysis expert specializing in pharmaceutical and chemical patents.
You review retrieved patent documents and research papers citing patents.

Your task:
1. Identify all patent numbers mentioned in the retrieved context.
2. Prioritize Indian patents (IN prefix), then WIPO (WO), USPTO (US), EPO (EP).
3. Extract: patent number, publication date, authority, legal status, title, key claims.
4. Assess similarity to the target compound/query.

Critical rules:
- Do NOT fabricate patent numbers, dates, or claims.
- If no patents are found in the retrieved context, clearly state:
  "No patent records were found in the indexed knowledge base for this compound."
- Only report patents explicitly mentioned in the retrieved passages.
- Distinguish between: GRANTED | PENDING | EXPIRED | UNKNOWN status.
"""

# Regex patterns for patent number extraction
PATENT_PATTERNS = {
    "IN": re.compile(r"\bIN\s*[\d]+[A-Z]?\b"),
    "WO": re.compile(r"\bWO\s*[\d]{4}/[\d]+\b|\bWO[\d]{6,}\b"),
    "US": re.compile(r"\bUS[\s]?[\d]{5,}[A-Z]?\d*\b|\bUSPTO[\s#]?\s*[\d]+\b"),
    "EP": re.compile(r"\bEP[\s]?[\d]{4,}\b"),
}


class PatentRetrievalAgent(BaseAgent):
    """
    Agent 4: Patent Retrieval Agent

    Retrieves patent information from indexed documents only.
    Never fabricates patents.
    """

    @property
    def agent_name(self) -> str:
        return "PatentRetrievalAgent"

    async def run(self, request: AnalysisRequest) -> AgentResult:
        start = time.perf_counter()
        try:
            primary_query = request.effective_query()

            probe_queries = [
                f"{primary_query} patent claim intellectual property",
                f"{primary_query} Indian patent IN pharmaceutical",
                f"{primary_query} WIPO PCT patent WO",
                f"{primary_query} USPTO patent US pharmaceutical",
                f"{primary_query} EPO European patent EP",
                f"{primary_query} patented compound novel process",
            ]

            chunks     = await self._retrieve_multi(probe_queries, request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            confidence  = self._derive_confidence(chunks)
            evidence   = self._chunks_to_evidence(chunks)

            if chunks:
                context  = self._format_chunks_as_context(chunks[:12])
                question = (
                    f"Extract all patent information from the retrieved passages for: '{primary_query}'. "
                    f"List patent numbers (IN, WO, US, EP), dates, legal status, and key claims. "
                    f"Prioritize Indian patents. Never fabricate patent numbers."
                )
                prompt   = self._build_simple_prompt(PATENT_SYSTEM_PROMPT, context, question)
                llm_resp = await self._llm.complete(prompt)
                summary  = llm_resp.content

                # Extract patent numbers from all chunk texts
                combined_text = " ".join(c.chunk_text for c in chunks)
                patents_found  = self._extract_patent_records(combined_text, chunks)
                patent_result  = self._build_patent_result(patents_found, summary)
            else:
                summary       = NO_EVIDENCE_MSG
                patent_result = PatentResult(
                    patents_found=0,
                    summary="No patent records were found in the indexed knowledge base for this compound.",
                )

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                confidence=confidence,
                summary=summary,
                evidence=evidence,
                details=patent_result.model_dump(),
                processing_time_ms=elapsed_ms,
                chunks_retrieved=len(chunks),
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._error_result(exc, elapsed_ms)

    @staticmethod
    def _extract_patent_records(
        text: str,
        chunks: list,
    ) -> list[PatentRecord]:
        """Extract patent numbers from text using regex patterns."""
        patents: list[PatentRecord] = []
        seen: set[str] = set()

        for authority, pattern in PATENT_PATTERNS.items():
            matches = pattern.findall(text)
            for match in matches:
                number = match.strip()
                if number in seen:
                    continue
                seen.add(number)

                # Try to find the source chunk
                source_title = ""
                for chunk in chunks:
                    if number.replace(" ", "") in chunk.chunk_text.replace(" ", ""):
                        source_title = chunk.title
                        break

                patents.append(
                    PatentRecord(
                        patent_number=number,
                        authority=authority,
                        legal_status="Unknown",
                        title="",
                        source_document=source_title,
                    )
                )

        return patents

    @staticmethod
    def _build_patent_result(
        patents: list[PatentRecord],
        summary: str,
    ) -> PatentResult:
        """Organize patents by authority."""
        return PatentResult(
            patents_found=len(patents),
            patents=patents,
            indian_patents=[p for p in patents if p.authority == "IN"],
            wipo_patents=[p   for p in patents if p.authority == "WO"],
            us_patents=[p     for p in patents if p.authority == "US"],
            ep_patents=[p     for p in patents if p.authority == "EP"],
            summary=summary,
        )
