"""
app/agents/literature_agent.py

Agent 1 – Literature Retrieval Agent

Responsibilities:
- Retrieve relevant scientific papers via semantic + multi-query search
- Extract key paragraphs and citations
- Rank evidence by relevance
- Return: relevant papers, key paragraphs, citation confidence
"""

from __future__ import annotations

import logging
import time
from collections import Counter

from app.agents.base_agent import NO_EVIDENCE_MSG, BaseAgent
from app.schemas.agents import AgentResult, Confidence, LiteratureResult
from app.schemas.analysis import AnalysisRequest

logger = logging.getLogger(__name__)


LITERATURE_SYSTEM_PROMPT = """\
You are a scientific literature analyst.
You receive retrieved passages from indexed research documents.
Your job is to:
1. Identify the most relevant papers
2. Extract key findings relevant to the query
3. Assess evidence quality and consistency
4. Note any conflicting findings explicitly

Rules:
- Answer ONLY from the retrieved context below.
- Never fabricate citations, authors, or findings.
- If the context is insufficient, state: "No supporting evidence was found in the indexed knowledge base."
- Format citations as: [Document Title, Page X, Section Y]
"""


class LiteratureRetrievalAgent(BaseAgent):
    """
    Agent 1: Literature Retrieval Agent

    Uses multi-query retrieval to probe different aspects of the query,
    merges results, and asks the LLM to synthesize key findings.
    """

    @property
    def agent_name(self) -> str:
        return "LiteratureRetrievalAgent"

    async def run(self, request: AnalysisRequest) -> AgentResult:
        start = time.perf_counter()
        try:
            primary_query = request.effective_query()

            # Build probe queries for richer multi-angle retrieval
            probe_queries = self._build_probe_queries(primary_query, request)

            # Retrieve across all probe queries
            chunks = await self._retrieve_multi(probe_queries, request)

            elapsed_ms = (time.perf_counter() - start) * 1000
            confidence  = self._derive_confidence(chunks)
            evidence    = self._chunks_to_evidence(chunks)

            # Deduplicate paper titles
            paper_titles: list[str] = []
            seen_titles:  set[str]  = set()
            for e in evidence:
                if e.title and e.title not in seen_titles:
                    seen_titles.add(e.title)
                    paper_titles.append(e.title)

            # Build top citations list
            top_citations = [
                f"{e.title}, p.{e.page}, §{e.section or 'N/A'}"
                for e in evidence[:10]
            ]

            # LLM synthesis
            if chunks:
                context = self._format_chunks_as_context(chunks[:15])
                question = (
                    f"Summarize the key scientific findings from these retrieved passages "
                    f"relevant to: '{primary_query}'. "
                    f"List the main papers, key findings, and any contradictions. "
                    f"Cite sources explicitly."
                )
                prompt = self._build_simple_prompt(LITERATURE_SYSTEM_PROMPT, context, question)
                llm_response = await self._llm.complete(prompt)
                summary = llm_response.content
            else:
                summary = NO_EVIDENCE_MSG

            lit_result = LiteratureResult(
                relevant_papers=paper_titles,
                key_paragraphs=evidence[:10],
                citation_count=len(evidence),
                top_citations=top_citations,
            )

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                confidence=confidence,
                summary=summary,
                evidence=evidence,
                details=lit_result.model_dump(),
                processing_time_ms=elapsed_ms,
                chunks_retrieved=len(chunks),
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._error_result(exc, elapsed_ms)

    def _build_probe_queries(
        self,
        primary_query: str,
        request: AnalysisRequest,
    ) -> list[str]:
        """Build multiple retrieval queries for comprehensive coverage."""
        queries = [primary_query]
        if request.molecule:
            mol = request.molecule
            if mol.name:
                queries.append(f"{mol.name} synthesis pharmacology mechanism")
                queries.append(f"{mol.name} clinical study research")
            if mol.smiles:
                queries.append(f"SMILES {mol.smiles} chemical properties")
        return list(dict.fromkeys(queries))  # deduplicate preserving order
