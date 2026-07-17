"""
app/agents/__init__.py

Agent registry – provides a centralized place to import and discover all agents.

Design: The orchestrator imports from here. Adding a new agent only requires
adding it to this module — the orchestrator does not need to change.
"""

from __future__ import annotations

from app.agents.base_agent                  import BaseAgent
from app.agents.chemical_availability_agent import ChemicalAvailabilityAgent
from app.agents.drug_likeness_agent         import DrugLikenessAgent
from app.agents.final_reasoning_agent       import FinalReasoningAgent
from app.agents.literature_agent            import LiteratureRetrievalAgent
from app.agents.novelty_agent               import NoveltyAgent
from app.agents.patent_agent                import PatentRetrievalAgent
from app.agents.safety_agent                import SafetyAgent
from app.agents.summarization_agent         import ResearchSummarizationAgent
from app.agents.synthetic_route_agent       import SyntheticRouteAgent
from app.agents.toxicity_agent              import ToxicityAgent

__all__ = [
    "BaseAgent",
    "LiteratureRetrievalAgent",
    "SyntheticRouteAgent",
    "ChemicalAvailabilityAgent",
    "PatentRetrievalAgent",
    "ToxicityAgent",
    "SafetyAgent",
    "DrugLikenessAgent",
    "NoveltyAgent",
    "ResearchSummarizationAgent",
    "FinalReasoningAgent",
]

# ── Parallel agent registry (Agents 1-8) ─────────────────────────────────────
# Adding a new parallel agent: add to PARALLEL_AGENTS list only.
PARALLEL_AGENTS: list[type[BaseAgent]] = [
    LiteratureRetrievalAgent,
    SyntheticRouteAgent,
    ChemicalAvailabilityAgent,
    PatentRetrievalAgent,
    ToxicityAgent,
    SafetyAgent,
    DrugLikenessAgent,
    NoveltyAgent,
]
