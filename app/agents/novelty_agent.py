"""
app/agents/novelty_agent.py

Agent 8 – Novelty Agent

Responsibilities:
- Compare retrieved literature to determine if similar molecules exist
- Compute Tanimoto similarity against retrieved SMILES (if available)
- Compute novelty score
- Return: similarity, novelty score, closest references, supporting evidence
"""

from __future__ import annotations

import logging
import re
import time

from app.agents.base_agent import NO_EVIDENCE_MSG, BaseAgent
from app.chemistry.rdkit_utils import compute_similarity, extract_scaffold
from app.schemas.agents import (
    AgentResult,
    Confidence,
    NoveltyResult,
    SimilarMolecule,
)
from app.schemas.analysis import AnalysisRequest

logger = logging.getLogger(__name__)


NOVELTY_SYSTEM_PROMPT = """\
You are a medicinal chemistry novelty analyst.
You compare the target compound against retrieved literature to assess structural novelty.

Your task:
1. Identify similar compounds mentioned in the retrieved passages.
2. Assess structural similarity (qualitative or quantitative if data is available).
3. Compute a novelty score (0 = identical to known compound, 1 = fully novel).
4. Identify the closest reference compound.
5. State whether the compound appears novel relative to the indexed literature.

Rules:
- Do NOT fabricate similar compounds or similarity scores.
- Only report compounds explicitly mentioned in the retrieved context.
- Novelty score should reflect evidence: if many similar compounds exist, novelty is low.
- Cite all references.
- Clearly distinguish: RETRIEVED EVIDENCE vs COMPUTATIONAL SIMILARITY.
"""

# Simple SMILES extractor (grabs tokens that look like SMILES)
SMILES_PATTERN = re.compile(
    r"(?<!\w)([A-Z][a-z]?(?:(?:\(|\[)?[CNOSPFIBrClc\(\)\[\]=\-\+\#\@\.0-9]+)+)(?!\w)"
)


class NoveltyAgent(BaseAgent):
    """
    Agent 8: Novelty Agent

    Uses both retrieval and RDKit Tanimoto similarity to assess novelty.
    """

    @property
    def agent_name(self) -> str:
        return "NoveltyAgent"

    async def run(self, request: AnalysisRequest) -> AgentResult:
        start = time.perf_counter()
        try:
            primary_query = request.effective_query()
            target_smiles = request.molecule.smiles if request.molecule else None
            mol_name      = request.molecule.name   if request.molecule else primary_query

            probe_queries = [
                f"{mol_name} analogues similar compounds structural comparison",
                f"{mol_name} novelty patented known compound scaffold",
                f"{mol_name} prior art related structures literature",
                f"{mol_name} derivative class chemical family",
            ]

            chunks     = await self._retrieve_multi(probe_queries, request)
            elapsed_ms  = (time.perf_counter() - start) * 1000
            confidence   = self._derive_confidence(chunks)
            evidence    = self._chunks_to_evidence(chunks)

            # Extract SMILES from retrieved chunks and compute Tanimoto
            similar_molecules: list[SimilarMolecule] = []
            if target_smiles and chunks:
                similar_molecules = self._find_similar(target_smiles, chunks)

            # LLM synthesis
            if chunks:
                context  = self._format_chunks_as_context(chunks[:10])
                sim_text = self._format_similarity_table(similar_molecules)
                question = (
                    f"Assess the structural novelty of '{mol_name}' based on the retrieved literature. "
                    f"Similarity results:\n{sim_text}\n"
                    f"State a novelty score (0=not novel, 1=fully novel). "
                    f"Identify the closest reference. Cite all sources."
                )
                prompt   = self._build_simple_prompt(NOVELTY_SYSTEM_PROMPT, context, question)
                llm_resp = await self._llm.complete(prompt)
                summary  = llm_resp.content
            else:
                summary = NO_EVIDENCE_MSG

            # Compute aggregate novelty score
            novelty_score = self._compute_novelty_score(similar_molecules)

            # Scaffold
            scaffold = ""
            if target_smiles:
                scaffold = extract_scaffold(target_smiles)

            closest = similar_molecules[0].name if similar_molecules else ""

            novelty_result = NoveltyResult(
                novelty_score=novelty_score,
                is_novel=novelty_score >= 0.70,
                similar_molecules=similar_molecules[:10],
                closest_reference=closest,
                structural_scaffold=scaffold,
                novelty_rationale=summary[:500] if summary else "",
            )

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                confidence=confidence,
                summary=summary,
                evidence=evidence,
                details=novelty_result.model_dump(),
                processing_time_ms=elapsed_ms,
                chunks_retrieved=len(chunks),
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._error_result(exc, elapsed_ms)

    def _find_similar(self, target_smiles: str, chunks: list) -> list[SimilarMolecule]:
        """Extract SMILES from chunk texts and compute Tanimoto similarity."""
        candidates: list[SimilarMolecule] = []
        seen: set[str] = set()

        for chunk in chunks:
            # Find candidate SMILES tokens in chunk text
            matches = SMILES_PATTERN.findall(chunk.chunk_text)
            for smiles_candidate in matches:
                if smiles_candidate in seen or smiles_candidate == target_smiles:
                    continue
                if len(smiles_candidate) < 5:   # Skip trivial tokens
                    continue
                seen.add(smiles_candidate)

                sim = compute_similarity(target_smiles, smiles_candidate)
                if sim > 0.10:  # Only keep meaningfully similar
                    candidates.append(
                        SimilarMolecule(
                            name=smiles_candidate[:80],
                            smiles=smiles_candidate,
                            similarity_score=sim,
                            source_document=chunk.title,
                            evidence_type="retrieved",
                        )
                    )

        # Sort by similarity descending
        candidates.sort(key=lambda m: m.similarity_score, reverse=True)
        return candidates[:15]

    @staticmethod
    def _compute_novelty_score(similar: list[SimilarMolecule]) -> float:
        """
        Compute novelty score (1 = fully novel, 0 = identical to known compound).

        Score = 1 - max_similarity_found (with some smoothing).
        """
        if not similar:
            return 0.85  # No similar found → probably novel, but uncertain
        max_sim = max(m.similarity_score for m in similar)
        return round(max(0.0, 1.0 - max_sim), 3)

    @staticmethod
    def _format_similarity_table(similar: list[SimilarMolecule]) -> str:
        if not similar:
            return "No structurally similar compounds extracted from retrieved chunks."
        rows = [
            f"- {m.name[:60]} | Tanimoto: {m.similarity_score:.3f} | Source: {m.source_document}"
            for m in similar[:10]
        ]
        return "\n".join(rows)
