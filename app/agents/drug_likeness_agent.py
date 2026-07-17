"""
app/agents/drug_likeness_agent.py

Agent 7 – Drug-likeness Agent

Responsibilities:
- Compute Lipinski, QED, MW, LogP, TPSA, HBD, HBA, RotBonds using RDKit
- Return drug-likeness summary and computed descriptors
- Uses SMILES from request molecule input
- Falls back gracefully if RDKit is unavailable or SMILES is invalid
"""

from __future__ import annotations

import logging
import time

from app.agents.base_agent import BaseAgent
from app.chemistry.rdkit_utils import MolecularDescriptors, compute_descriptors
from app.schemas.agents import (
    AgentResult,
    Confidence,
    DrugLikenessDescriptors,
    DrugLikenessResult,
)
from app.schemas.analysis import AnalysisRequest

logger = logging.getLogger(__name__)


DRUG_LIKENESS_SYSTEM_PROMPT = """\
You are a medicinal chemist expert in drug-likeness assessment.

You receive computed molecular descriptors and retrieved literature passages.
Your task:
1. Interpret the computed descriptors.
2. Assess drug-likeness (Lipinski Ro5, QED score, CNS penetrance if TPSA < 90).
3. Identify strengths and weaknesses.
4. Provide a concise drug-likeness summary.

Always state: "Descriptors computed using RDKit."
"""


class DrugLikenessAgent(BaseAgent):
    """
    Agent 7: Drug-likeness Agent

    Computes molecular descriptors using RDKit, then retrieves literature
    context to enrich the drug-likeness assessment.
    """

    @property
    def agent_name(self) -> str:
        return "DrugLikenessAgent"

    async def run(self, request: AnalysisRequest) -> AgentResult:
        start = time.perf_counter()
        try:
            primary_query = request.effective_query()
            smiles        = request.molecule.smiles if request.molecule else None
            mol_name      = request.molecule.name   if request.molecule else primary_query

            # ── Step 1: Compute descriptors ────────────────────────────────
            descriptors   = MolecularDescriptors()
            smiles_used   = ""
            compute_note  = ""

            if smiles:
                descriptors  = compute_descriptors(smiles)
                smiles_used  = smiles
                compute_note = (
                    "Descriptors computed using RDKit."
                    if descriptors.smiles_valid
                    else f"Invalid SMILES: '{smiles}'. Descriptor computation skipped."
                )
            else:
                compute_note = (
                    "No SMILES string provided. "
                    "Drug-likeness computation requires a valid SMILES input."
                )

            # ── Step 2: Retrieve literature context ────────────────────────
            probe_queries = [
                f"{mol_name} drug-likeness pharmacokinetics bioavailability",
                f"{mol_name} Lipinski rule of five oral absorption",
                f"{mol_name} ADMET permeability metabolic stability",
            ]
            chunks     = await self._retrieve_multi(probe_queries, request)
            evidence   = self._chunks_to_evidence(chunks)

            # ── Step 3: LLM synthesis ──────────────────────────────────────
            descriptor_text = self._descriptors_to_text(descriptors)
            context         = self._format_chunks_as_context(chunks[:8])
            question        = (
                f"Assess the drug-likeness of '{mol_name}' given these computed descriptors:\n"
                f"{descriptor_text}\n\n"
                f"Also consider any relevant literature context above. "
                f"Summarize drug-likeness, flag any Lipinski violations, and assess CNS penetrance potential."
            )
            prompt   = self._build_simple_prompt(DRUG_LIKENESS_SYSTEM_PROMPT, context, question)
            llm_resp = await self._llm.complete(prompt)
            summary  = llm_resp.content

            # ── Step 4: Build result ───────────────────────────────────────
            dl_descriptors = DrugLikenessDescriptors(
                molecular_weight=descriptors.molecular_weight,
                logp=descriptors.logp,
                tpsa=descriptors.tpsa,
                hbd=descriptors.hbd,
                hba=descriptors.hba,
                rotatable_bonds=descriptors.rotatable_bonds,
                qed=descriptors.qed,
                lipinski_violations=descriptors.lipinski_violations,
                passes_lipinski=descriptors.passes_lipinski,
                aromatic_rings=descriptors.aromatic_rings,
                molecular_formula=descriptors.molecular_formula,
                exact_mw=descriptors.exact_mw,
            )

            dl_result = DrugLikenessResult(
                descriptors=dl_descriptors,
                drug_likeness_summary=summary,
                computed_via="RDKit" if descriptors.smiles_valid else "N/A",
                smiles_valid=descriptors.smiles_valid,
                smiles_used=smiles_used,
                computation_note=compute_note,
            )

            # Confidence is based on SMILES validity + literature retrieval
            if descriptors.smiles_valid and chunks:
                confidence = Confidence.HIGH
            elif descriptors.smiles_valid:
                confidence = Confidence.MEDIUM
            elif chunks:
                confidence = Confidence.LOW
            else:
                confidence = Confidence.NONE

            elapsed_ms = (time.perf_counter() - start) * 1000

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                confidence=confidence,
                summary=summary,
                evidence=evidence,
                details=dl_result.model_dump(),
                processing_time_ms=elapsed_ms,
                chunks_retrieved=len(chunks),
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._error_result(exc, elapsed_ms)

    @staticmethod
    def _descriptors_to_text(d: MolecularDescriptors) -> str:
        """Format descriptors as readable text for the LLM."""
        if not d.smiles_valid:
            return "No valid SMILES provided. Descriptor computation not performed."
        lines = [
            f"• Molecular Formula:   {d.molecular_formula or 'N/A'}",
            f"• Molecular Weight:    {d.molecular_weight or 'N/A'} Da",
            f"• Exact MW:            {d.exact_mw or 'N/A'} Da",
            f"• LogP:                {d.logp or 'N/A'}",
            f"• TPSA:                {d.tpsa or 'N/A'} Å²",
            f"• H-Bond Donors:       {d.hbd or 'N/A'}",
            f"• H-Bond Acceptors:    {d.hba or 'N/A'}",
            f"• Rotatable Bonds:     {d.rotatable_bonds or 'N/A'}",
            f"• Aromatic Rings:      {d.aromatic_rings or 'N/A'}",
            f"• QED Score:           {d.qed or 'N/A'} (0=poor, 1=ideal)",
            f"• Lipinski Violations: {d.lipinski_violations or 0}",
            f"• Passes Lipinski Ro5: {d.passes_lipinski}",
        ]
        return "\n".join(lines)
