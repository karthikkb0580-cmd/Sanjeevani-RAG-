"""
app/schemas/analysis.py

Top-level request / response schemas for the multi-agent analysis endpoint.

POST /api/v2/analyze  →  AnalysisRequest  →  FinalAnalysisReport
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.agents import (
    AgentResult,
    Confidence,
    DrugLikenessResult,
    LiteratureResult,
    NoveltyResult,
    PatentResult,
    ResearchSummaryResult,
    SafetyResult,
    SyntheticRouteResult,
    ToxicityResult,
    ChemicalAvailabilityResult,
)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class MoleculeInput(BaseModel):
    """Optional structured molecule input."""
    name:   str | None = Field(default=None, description="IUPAC or common name")
    smiles: str | None = Field(default=None, description="SMILES string")
    inchi:  str | None = Field(default=None, description="InChI string")
    cas:    str | None = Field(default=None, description="CAS registry number")


class AnalysisRequest(BaseModel):
    """
    Multi-agent analysis request.

    Either `query` or `molecule` (or both) must be provided.
    """
    query:      str | None        = Field(
        default=None,
        max_length=4000,
        description="Free-text query, paper title, or research question",
    )
    molecule:   MoleculeInput | None = Field(
        default=None,
        description="Structured molecule input",
    )
    # Retrieval parameters (override defaults from settings)
    top_k:                  int   = Field(default=10, ge=1, le=50)
    similarity_threshold:   float = Field(default=0.60, ge=0.0, le=1.0)
    use_mmr:                bool  = Field(default=True)
    mmr_lambda:             float = Field(default=0.5, ge=0.0, le=1.0)
    # Control which agents run
    agents_enabled:         list[str] | None = Field(
        default=None,
        description="If provided, run only these agents (by name). Run all if None.",
    )
    stream:                 bool  = Field(
        default=False,
        description="Stream the final report over SSE",
    )

    def effective_query(self) -> str:
        """Derive the primary query string used across all agents."""
        parts: list[str] = []
        if self.query:
            parts.append(self.query)
        if self.molecule:
            if self.molecule.name:
                parts.append(self.molecule.name)
            if self.molecule.smiles:
                parts.append(f"SMILES: {self.molecule.smiles}")
            if self.molecule.cas:
                parts.append(f"CAS: {self.molecule.cas}")
        if not parts:
            raise ValueError("Either 'query' or 'molecule' must be provided")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Confidence Assessment Section
# ---------------------------------------------------------------------------

class ConfidenceAssessment(BaseModel):
    """Per-section confidence assessments for the final report."""
    literature:             Confidence = Confidence.NONE
    synthetic_route:        Confidence = Confidence.NONE
    chemical_availability:  Confidence = Confidence.NONE
    patent_landscape:       Confidence = Confidence.NONE
    toxicity:               Confidence = Confidence.NONE
    safety:                 Confidence = Confidence.NONE
    drug_likeness:          Confidence = Confidence.NONE
    novelty:                Confidence = Confidence.NONE
    overall:                Confidence = Confidence.NONE

    def overall_from_agents(self, results: dict[str, AgentResult]) -> None:
        """Auto-populate from agent results and compute overall."""
        mapping = {
            "LiteratureRetrievalAgent":    "literature",
            "SyntheticRouteAgent":         "synthetic_route",
            "ChemicalAvailabilityAgent":   "chemical_availability",
            "PatentRetrievalAgent":        "patent_landscape",
            "ToxicityAgent":               "toxicity",
            "SafetyAgent":                 "safety",
            "DrugLikenessAgent":           "drug_likeness",
            "NoveltyAgent":                "novelty",
        }
        confidences: list[Confidence] = []
        for agent_name, field in mapping.items():
            result = results.get(agent_name)
            if result:
                setattr(self, field, result.confidence)
                confidences.append(result.confidence)

        # Overall = most common / pessimistic aggregation
        _order = {
            Confidence.HIGH: 3, Confidence.MEDIUM: 2,
            Confidence.LOW: 1, Confidence.NONE: 0,
        }
        if confidences:
            avg = sum(_order[c] for c in confidences) / len(confidences)
            if avg >= 2.5:
                self.overall = Confidence.HIGH
            elif avg >= 1.5:
                self.overall = Confidence.MEDIUM
            elif avg >= 0.5:
                self.overall = Confidence.LOW
            else:
                self.overall = Confidence.NONE


# ---------------------------------------------------------------------------
# Final Analysis Report
# ---------------------------------------------------------------------------

class FinalAnalysisReport(BaseModel):
    """
    Structured multi-agent RAG analysis report.

    Maps directly to the 13 required output sections.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    request_id:         str             = Field(...)
    query:              str             = Field(...)
    molecule_input:     MoleculeInput | None = Field(default=None)

    # ── Sections ──────────────────────────────────────────────────────────────
    # 1. Executive Summary
    executive_summary:              str     = Field(default="")

    # 2. Molecule Overview
    molecule_overview:              str     = Field(default="")

    # 3. Literature Findings
    literature_findings:            LiteratureResult | None     = Field(default=None)

    # 4. Synthetic Route Summary
    synthetic_route_summary:        SyntheticRouteResult | None = Field(default=None)

    # 5. Reagent Availability Status
    reagent_availability_status:    ChemicalAvailabilityResult | None = Field(default=None)

    # 6. Patent Landscape
    patent_landscape:               PatentResult | None         = Field(default=None)

    # 7. Toxicity Evidence Summary
    toxicity_evidence_summary:      ToxicityResult | None       = Field(default=None)

    # 8. Laboratory Safety Summary
    laboratory_safety_summary:      SafetyResult | None         = Field(default=None)

    # 9. Drug-likeness Assessment
    drug_likeness_assessment:       DrugLikenessResult | None   = Field(default=None)

    # 10. Novelty Assessment
    novelty_assessment:             NoveltyResult | None        = Field(default=None)

    # 11. Research Gaps
    research_gaps:                  list[str]                   = Field(default_factory=list)

    # 12. Confidence Assessment
    confidence_assessment:          ConfidenceAssessment        = Field(
        default_factory=ConfidenceAssessment
    )

    # 13. References
    references:                     list[str]                   = Field(default_factory=list)

    # ── Metadata ─────────────────────────────────────────────────────────────
    research_summary_narrative:     str         = Field(default="")
    final_reasoning_narrative:      str         = Field(default="")
    agent_results:                  dict[str, AgentResult] = Field(default_factory=dict)
    agents_succeeded:               int         = Field(default=0)
    agents_failed:                  int         = Field(default=0)
    total_processing_time_ms:       float       = Field(default=0.0)
    llm_model:                      str         = Field(default="")
    embedding_model:                str         = Field(default="")
    timestamp:                      datetime    = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
