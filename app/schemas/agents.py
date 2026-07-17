"""
app/schemas/agents.py

Pydantic v2 models for all multi-agent RAG components.

Defines:
  - AgentResult       : Universal output envelope for every agent
  - Confidence        : Evidence quality enum
  - EvidenceItem      : A single retrieved evidence piece with citation
  - Per-agent output models (LiteratureResult, SyntheticRouteResult, ...)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Confidence Levels
# ---------------------------------------------------------------------------

class Confidence(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"
    NONE   = "none"  # No evidence retrieved


# ---------------------------------------------------------------------------
# Evidence Item (universal citation unit)
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    """A single retrieved evidence piece with full provenance."""
    document_id:      str   = Field(..., description="Source document UUID")
    title:            str   = Field(..., description="Document / paper title")
    page:             int   = Field(default=0, description="Page number")
    section:          str   = Field(default="", description="Section heading")
    chunk_text:       str   = Field(..., description="Verbatim retrieved passage")
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_type:    str   = Field(
        default="retrieved",
        description="retrieved | predicted | heuristic | generated",
    )


# ---------------------------------------------------------------------------
# Universal Agent Result envelope
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    """
    Structured output produced by every agent.

    Always populated:
    - agent_name    : Identifies which agent produced this
    - success       : False if the agent failed
    - confidence    : Based on retrieval quality and evidence consistency

    Populated on success:
    - summary       : Short plain-text summary of findings
    - evidence      : List of retrieved evidence items with citations
    - details       : Agent-specific structured payload (varies per agent)
    - error         : Error message when success=False
    """
    agent_name:       str                 = Field(...)
    success:          bool                = Field(default=True)
    confidence:       Confidence          = Field(default=Confidence.LOW)
    summary:          str                 = Field(default="")
    evidence:         list[EvidenceItem]  = Field(default_factory=list)
    details:          dict[str, Any]      = Field(default_factory=dict)
    error:            str | None          = Field(default=None)
    processing_time_ms: float            = Field(default=0.0)
    chunks_retrieved: int                = Field(default=0)
    timestamp:        datetime           = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Agent 1 – Literature Retrieval
# ---------------------------------------------------------------------------

class LiteratureResult(BaseModel):
    """Structured payload for the Literature Retrieval Agent."""
    relevant_papers:    list[str]         = Field(default_factory=list, description="Paper titles")
    key_paragraphs:     list[EvidenceItem] = Field(default_factory=list)
    citation_count:     int               = Field(default=0)
    top_citations:      list[str]         = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent 2 – Synthetic Route
# ---------------------------------------------------------------------------

class SyntheticStep(BaseModel):
    step_number:    int
    description:    str
    reagents:       list[str] = Field(default_factory=list)
    conditions:     str       = Field(default="")
    citation:       str       = Field(default="")
    is_literature_derived: bool = Field(default=True)


class SyntheticRouteResult(BaseModel):
    """Structured payload for the Synthetic Route Agent."""
    routes_found:           int                  = Field(default=0)
    reaction_sequence:      list[SyntheticStep]  = Field(default_factory=list)
    reported_reagents:      list[str]            = Field(default_factory=list)
    number_of_steps:        int | None           = Field(default=None)
    estimated_complexity:   str                  = Field(
        default="Unknown",
        description="Simple | Moderate | Complex | Highly Complex",
    )
    starting_materials:     list[str]            = Field(default_factory=list)
    supporting_citations:   list[str]            = Field(default_factory=list)
    is_literature_derived:  bool                 = Field(default=True)


# ---------------------------------------------------------------------------
# Agent 3 – Chemical Availability
# ---------------------------------------------------------------------------

class AvailabilityStatus(str, Enum):
    AVAILABLE       = "available"
    LIMITED         = "limited"
    NOT_AVAILABLE   = "not_available"
    UNVERIFIED      = "unverified"


class ChemicalAvailabilityResult(BaseModel):
    """Structured payload for the Chemical Availability Agent."""
    availability_status:  AvailabilityStatus = Field(default=AvailabilityStatus.UNVERIFIED)
    possible_suppliers:   list[str]          = Field(default_factory=list)
    evidence_source:      str                = Field(default="No configured data source")
    notes:                str                = Field(default="")


# ---------------------------------------------------------------------------
# Agent 4 – Patent Retrieval
# ---------------------------------------------------------------------------

class PatentRecord(BaseModel):
    patent_number:    str
    publication_date: str = Field(default="")
    authority:        str = Field(default="", description="IN | WO | US | EP")
    legal_status:     str = Field(default="")
    title:            str = Field(default="")
    relevant_claims:  list[str] = Field(default_factory=list)
    similarity_score: float     = Field(default=0.0)
    source_document:  str       = Field(default="")


class PatentResult(BaseModel):
    """Structured payload for the Patent Retrieval Agent."""
    patents_found:    int               = Field(default=0)
    patents:          list[PatentRecord] = Field(default_factory=list)
    indian_patents:   list[PatentRecord] = Field(default_factory=list)
    wipo_patents:     list[PatentRecord] = Field(default_factory=list)
    us_patents:       list[PatentRecord] = Field(default_factory=list)
    ep_patents:       list[PatentRecord] = Field(default_factory=list)
    summary:          str               = Field(default="")


# ---------------------------------------------------------------------------
# Agent 5 – Toxicity
# ---------------------------------------------------------------------------

class ToxicityClassification(BaseModel):
    classification: str  = Field(description="e.g. GHS Category")
    hazard_code:    str  = Field(default="")
    source:         str  = Field(default="")
    is_predicted:   bool = Field(default=False)


class ToxicityResult(BaseModel):
    """Structured payload for the Toxicity Information Agent."""
    known_hazards:          list[str]                   = Field(default_factory=list)
    exposure_considerations: list[str]                  = Field(default_factory=list)
    classifications:        list[ToxicityClassification] = Field(default_factory=list)
    relevant_citations:     list[str]                   = Field(default_factory=list)
    data_gap_note:          str                         = Field(default="")
    disclaimer:             str = Field(
        default=(
            "This information is retrieved from scientific literature for "
            "research purposes only. It does not constitute medical or "
            "regulatory advice."
        )
    )


# ---------------------------------------------------------------------------
# Agent 6 – Safety
# ---------------------------------------------------------------------------

class SafetyResult(BaseModel):
    """Structured payload for the Safety Information Agent."""
    handling_precautions:       list[str] = Field(default_factory=list)
    storage_recommendations:    list[str] = Field(default_factory=list)
    ppe_recommendations:        list[str] = Field(default_factory=list)
    disposal_notes:             list[str] = Field(default_factory=list)
    supporting_references:      list[str] = Field(default_factory=list)
    disclaimer:                 str = Field(
        default=(
            "Laboratory safety guidance retrieved from indexed references. "
            "Always consult institutional EHS protocols before handling chemicals."
        )
    )


# ---------------------------------------------------------------------------
# Agent 7 – Drug-likeness
# ---------------------------------------------------------------------------

class DrugLikenessDescriptors(BaseModel):
    molecular_weight:           float | None = None
    logp:                       float | None = None
    tpsa:                       float | None = None
    hbd:                        int   | None = None  # H-bond donors
    hba:                        int   | None = None  # H-bond acceptors
    rotatable_bonds:            int   | None = None
    qed:                        float | None = None
    lipinski_violations:        int   | None = None
    passes_lipinski:            bool  | None = None
    aromatic_rings:             int   | None = None
    molecular_formula:          str   | None = None
    exact_mw:                   float | None = None


class DrugLikenessResult(BaseModel):
    """Structured payload for the Drug-likeness Agent."""
    descriptors:        DrugLikenessDescriptors = Field(
        default_factory=DrugLikenessDescriptors
    )
    drug_likeness_summary: str  = Field(default="")
    computed_via:       str     = Field(default="RDKit")
    smiles_valid:       bool    = Field(default=False)
    smiles_used:        str     = Field(default="")
    computation_note:   str     = Field(default="")


# ---------------------------------------------------------------------------
# Agent 8 – Novelty
# ---------------------------------------------------------------------------

class SimilarMolecule(BaseModel):
    name:             str
    smiles:           str       = Field(default="")
    similarity_score: float     = Field(default=0.0)
    source_document:  str       = Field(default="")
    evidence_type:    str       = Field(default="retrieved")


class NoveltyResult(BaseModel):
    """Structured payload for the Novelty Agent."""
    novelty_score:        float            = Field(default=0.0, ge=0.0, le=1.0)
    is_novel:             bool             = Field(default=False)
    similar_molecules:    list[SimilarMolecule] = Field(default_factory=list)
    closest_reference:    str             = Field(default="")
    structural_scaffold:  str             = Field(default="")
    novelty_rationale:    str             = Field(default="")


# ---------------------------------------------------------------------------
# Agent 9 – Research Summarization
# ---------------------------------------------------------------------------

class ResearchSummaryResult(BaseModel):
    """Structured payload for the Research Summarization Agent."""
    key_findings:       list[str] = Field(default_factory=list)
    limitations:        list[str] = Field(default_factory=list)
    consensus_points:   list[str] = Field(default_factory=list)
    research_gaps:      list[str] = Field(default_factory=list)
    paper_count:        int       = Field(default=0)
    synthesis_narrative: str      = Field(default="")
