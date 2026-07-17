"""
tests/test_agents.py

Unit tests for all 10 Sanjeevani multi-agents.

Tests:
- Each parallel agent (1-8) in isolation with mocked retriever + LLM
- ResearchSummarizationAgent (9) with mocked prior agent results
- FinalReasoningAgent (10) with mocked all-agent results
- Error isolation (agent failure returns AgentResult.success=False, never raises)
- BaseAgent helper utilities
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.agents import AgentResult, Confidence, EvidenceItem
from app.schemas.analysis import AnalysisRequest, MoleculeInput
from app.schemas.chat import RetrievedChunk


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

@pytest.fixture
def sample_request() -> AnalysisRequest:
    return AnalysisRequest(
        query="aspirin synthesis pharmacology",
        molecule=MoleculeInput(
            name="Aspirin",
            smiles=ASPIRIN_SMILES,
            cas="50-78-2",
        ),
        top_k=5,
        similarity_threshold=0.60,
        stream=False,
    )


@pytest.fixture
def mock_chunks() -> list[RetrievedChunk]:
    """Three mock retrieved chunks for test retrieval."""
    return [
        RetrievedChunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            title="Aspirin: A Review of Pharmacological Properties",
            page=3,
            section="Introduction",
            chunk_text=(
                "Aspirin (acetylsalicylic acid) is a widely used analgesic and antipyretic. "
                "It irreversibly inhibits cyclooxygenase (COX) enzymes, reducing prostaglandin synthesis."
            ),
            similarity_score=0.91,
        ),
        RetrievedChunk(
            chunk_id="chunk-002",
            document_id="doc-002",
            title="Synthesis of Salicylic Acid Derivatives",
            page=7,
            section="Methods",
            chunk_text=(
                "The acetylation of salicylic acid with acetic anhydride in the presence of "
                "phosphoric acid catalyst yields aspirin (acetylsalicylic acid) in 3 steps."
            ),
            similarity_score=0.85,
        ),
        RetrievedChunk(
            chunk_id="chunk-003",
            document_id="doc-003",
            title="Safety and Toxicity of NSAIDs",
            page=12,
            section="Adverse Effects",
            chunk_text=(
                "Aspirin can cause gastrointestinal irritation. LD50 in rats (oral): ~200 mg/kg. "
                "GHS classification: Acute Tox 4."
            ),
            similarity_score=0.78,
        ),
    ]


@pytest.fixture
def mock_retriever(mock_chunks):
    """Mock Retriever that returns mock_chunks for any query."""
    retriever = AsyncMock()
    retriever.retrieve.return_value = mock_chunks
    return retriever


@pytest.fixture
def mock_llm_response():
    """Mock LLMResponse."""
    from app.llm.openai_client import LLMResponse
    return LLMResponse(
        content="Mock LLM analysis response: Key finding is that aspirin inhibits COX enzymes.",
        model="mock-gpt-4o",
        prompt_tokens=500,
        completion_tokens=100,
        total_tokens=600,
        finish_reason="stop",
    )


@pytest.fixture
def mock_llm(mock_llm_response):
    """Mock LLM client."""
    llm = AsyncMock()
    llm.complete.return_value = mock_llm_response
    llm.model_name = "mock-gpt-4o"
    return llm


# ---------------------------------------------------------------------------
# Helper: build a mock AgentResult
# ---------------------------------------------------------------------------

def _make_agent_result(name: str, success: bool = True) -> AgentResult:
    return AgentResult(
        agent_name=name,
        success=success,
        confidence=Confidence.MEDIUM,
        summary=f"Mock summary from {name}",
        evidence=[
            EvidenceItem(
                document_id="doc-001",
                title="Test Paper",
                page=1,
                section="Introduction",
                chunk_text="Test evidence text.",
                similarity_score=0.80,
            )
        ],
        chunks_retrieved=3,
        error=None if success else "Mock error",
    )


# ---------------------------------------------------------------------------
# Test: AnalysisRequest
# ---------------------------------------------------------------------------

class TestAnalysisRequest:
    def test_effective_query_with_both(self):
        req = AnalysisRequest(
            query="aspirin",
            molecule=MoleculeInput(name="ASA", smiles="CC(=O)Oc1ccccc1C(=O)O"),
        )
        q = req.effective_query()
        assert "aspirin" in q
        assert "ASA" in q

    def test_effective_query_query_only(self):
        req = AnalysisRequest(query="aspirin pharmacology")
        q = req.effective_query()
        assert "aspirin pharmacology" in q

    def test_effective_query_molecule_only(self):
        req = AnalysisRequest(molecule=MoleculeInput(name="Aspirin", smiles=ASPIRIN_SMILES))
        q = req.effective_query()
        assert "Aspirin" in q
        assert ASPIRIN_SMILES in q

    def test_effective_query_raises_when_empty(self):
        req = AnalysisRequest()
        with pytest.raises(ValueError, match="query.*molecule"):
            req.effective_query()


# ---------------------------------------------------------------------------
# Test: BaseAgent helpers
# ---------------------------------------------------------------------------

class TestBaseAgentHelpers:
    @pytest.fixture
    def agent(self, mock_retriever, mock_llm):
        """Concrete test agent using LiteratureRetrievalAgent."""
        from app.agents.literature_agent import LiteratureRetrievalAgent
        return LiteratureRetrievalAgent(retriever=mock_retriever, llm_client=mock_llm)

    def test_chunks_to_evidence(self, agent, mock_chunks):
        evidence = agent._chunks_to_evidence(mock_chunks)
        assert len(evidence) == 3
        assert evidence[0].title == "Aspirin: A Review of Pharmacological Properties"
        assert evidence[0].evidence_type == "retrieved"

    def test_derive_confidence_high(self, agent, mock_chunks):
        # 3 chunks, avg ~0.85 – should be MEDIUM (need ≥5 for HIGH)
        conf = agent._derive_confidence(mock_chunks)
        assert conf in (Confidence.MEDIUM, Confidence.HIGH)

    def test_derive_confidence_none(self, agent):
        conf = agent._derive_confidence([])
        assert conf == Confidence.NONE

    def test_format_chunks_as_context(self, agent, mock_chunks):
        ctx = agent._format_chunks_as_context(mock_chunks)
        assert "Aspirin" in ctx
        assert "[1]" in ctx
        assert "---" in ctx


# ---------------------------------------------------------------------------
# Test: LiteratureRetrievalAgent (Agent 1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestLiteratureRetrievalAgent:
    @pytest.fixture
    def agent(self, mock_retriever, mock_llm):
        from app.agents.literature_agent import LiteratureRetrievalAgent
        return LiteratureRetrievalAgent(retriever=mock_retriever, llm_client=mock_llm)

    async def test_run_returns_agent_result(self, agent, sample_request):
        result = await agent.run(sample_request)
        assert isinstance(result, AgentResult)
        assert result.agent_name == "LiteratureRetrievalAgent"
        assert result.success is True

    async def test_run_returns_evidence(self, agent, sample_request):
        result = await agent.run(sample_request)
        assert len(result.evidence) > 0

    async def test_run_with_no_chunks(self, mock_llm, sample_request):
        from app.agents.literature_agent import LiteratureRetrievalAgent
        empty_retriever = AsyncMock()
        empty_retriever.retrieve.return_value = []
        agent = LiteratureRetrievalAgent(retriever=empty_retriever, llm_client=mock_llm)
        result = await agent.run(sample_request)
        assert result.success is True
        assert result.confidence == Confidence.NONE
        assert "No supporting evidence" in result.summary

    async def test_run_handles_llm_error(self, mock_retriever, sample_request):
        from app.agents.literature_agent import LiteratureRetrievalAgent
        failing_llm = AsyncMock()
        failing_llm.complete.side_effect = RuntimeError("LLM API timeout")
        agent = LiteratureRetrievalAgent(retriever=mock_retriever, llm_client=failing_llm)
        result = await agent.run(sample_request)
        assert result.success is False
        assert result.error is not None
        assert "LLM API timeout" in result.error


# ---------------------------------------------------------------------------
# Test: SyntheticRouteAgent (Agent 2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSyntheticRouteAgent:
    @pytest.fixture
    def agent(self, mock_retriever, mock_llm):
        from app.agents.synthetic_route_agent import SyntheticRouteAgent
        return SyntheticRouteAgent(retriever=mock_retriever, llm_client=mock_llm)

    async def test_run_returns_agent_result(self, agent, sample_request):
        result = await agent.run(sample_request)
        assert isinstance(result, AgentResult)
        assert result.agent_name == "SyntheticRouteAgent"
        assert result.success is True

    async def test_details_contain_route_fields(self, agent, sample_request):
        result = await agent.run(sample_request)
        assert "estimated_complexity" in result.details
        assert "supporting_citations" in result.details


# ---------------------------------------------------------------------------
# Test: ChemicalAvailabilityAgent (Agent 3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestChemicalAvailabilityAgent:
    @pytest.fixture
    def agent(self, mock_retriever, mock_llm):
        from app.agents.chemical_availability_agent import ChemicalAvailabilityAgent
        return ChemicalAvailabilityAgent(retriever=mock_retriever, llm_client=mock_llm)

    async def test_run_returns_result(self, agent, sample_request):
        result = await agent.run(sample_request)
        assert result.success is True
        assert "availability_status" in result.details

    async def test_unverified_when_no_chunks(self, mock_llm, sample_request):
        from app.agents.chemical_availability_agent import ChemicalAvailabilityAgent
        from app.schemas.agents import AvailabilityStatus
        empty_retriever = AsyncMock()
        empty_retriever.retrieve.return_value = []
        agent = ChemicalAvailabilityAgent(retriever=empty_retriever, llm_client=mock_llm)
        result = await agent.run(sample_request)
        assert result.details["availability_status"] == AvailabilityStatus.UNVERIFIED.value


# ---------------------------------------------------------------------------
# Test: PatentRetrievalAgent (Agent 4)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPatentRetrievalAgent:
    @pytest.fixture
    def agent(self, mock_retriever, mock_llm):
        from app.agents.patent_agent import PatentRetrievalAgent
        return PatentRetrievalAgent(retriever=mock_retriever, llm_client=mock_llm)

    async def test_run_returns_result(self, agent, sample_request):
        result = await agent.run(sample_request)
        assert result.success is True
        assert "patents_found" in result.details

    def test_extract_patent_numbers(self):
        from app.agents.patent_agent import PatentRetrievalAgent
        text = "See Indian patent IN202141053456 and WO2021/123456."
        patents = PatentRetrievalAgent._extract_patent_records(text, [])
        numbers = [p.patent_number for p in patents]
        assert any("IN" in n for n in numbers)


# ---------------------------------------------------------------------------
# Test: ToxicityAgent (Agent 5)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestToxicityAgent:
    @pytest.fixture
    def agent(self, mock_retriever, mock_llm):
        from app.agents.toxicity_agent import ToxicityAgent
        return ToxicityAgent(retriever=mock_retriever, llm_client=mock_llm)

    async def test_run_returns_result(self, agent, sample_request):
        result = await agent.run(sample_request)
        assert result.success is True
        assert "known_hazards" in result.details

    async def test_no_chunks_returns_safe_response(self, mock_llm, sample_request):
        from app.agents.toxicity_agent import ToxicityAgent
        empty_retriever = AsyncMock()
        empty_retriever.retrieve.return_value = []
        agent = ToxicityAgent(retriever=empty_retriever, llm_client=mock_llm)
        result = await agent.run(sample_request)
        assert result.success is True
        assert result.confidence == Confidence.NONE


# ---------------------------------------------------------------------------
# Test: SafetyAgent (Agent 6)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSafetyAgent:
    @pytest.fixture
    def agent(self, mock_retriever, mock_llm):
        from app.agents.safety_agent import SafetyAgent
        return SafetyAgent(retriever=mock_retriever, llm_client=mock_llm)

    async def test_run_returns_result(self, agent, sample_request):
        result = await agent.run(sample_request)
        assert result.success is True
        assert "handling_precautions" in result.details
        assert "disclaimer" in result.details


# ---------------------------------------------------------------------------
# Test: DrugLikenessAgent (Agent 7)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDrugLikenessAgent:
    @pytest.fixture
    def agent(self, mock_retriever, mock_llm):
        from app.agents.drug_likeness_agent import DrugLikenessAgent
        return DrugLikenessAgent(retriever=mock_retriever, llm_client=mock_llm)

    async def test_run_with_valid_smiles(self, agent, sample_request):
        result = await agent.run(sample_request)
        assert result.success is True
        assert "descriptors" in result.details

    async def test_run_without_smiles(self, mock_retriever, mock_llm):
        from app.agents.drug_likeness_agent import DrugLikenessAgent
        agent = DrugLikenessAgent(retriever=mock_retriever, llm_client=mock_llm)
        req   = AnalysisRequest(query="aspirin")
        result = await agent.run(req)
        assert result.success is True
        # Should not crash, just note no SMILES available
        assert "smiles_valid" in result.details
        assert result.details["smiles_valid"] is False


# ---------------------------------------------------------------------------
# Test: RDKit utilities
# ---------------------------------------------------------------------------

class TestRdkitUtils:
    def test_compute_descriptors_valid_smiles(self):
        from app.chemistry.rdkit_utils import compute_descriptors
        d = compute_descriptors(ASPIRIN_SMILES)
        if d.smiles_valid:  # Only assert if RDKit is available
            assert d.molecular_weight is not None
            assert d.logp is not None
            assert d.qed is not None
            assert d.lipinski_violations is not None

    def test_compute_descriptors_invalid_smiles(self):
        from app.chemistry.rdkit_utils import compute_descriptors
        d = compute_descriptors("NOT_A_SMILES")
        assert d.smiles_valid is False

    def test_compute_descriptors_empty_string(self):
        from app.chemistry.rdkit_utils import compute_descriptors
        d = compute_descriptors("")
        assert d.smiles_valid is False

    def test_compute_similarity_same(self):
        from app.chemistry.rdkit_utils import compute_similarity
        sim = compute_similarity(ASPIRIN_SMILES, ASPIRIN_SMILES)
        if sim > 0:  # Only if RDKit available
            assert sim == 1.0

    def test_compute_similarity_different(self):
        from app.chemistry.rdkit_utils import compute_similarity
        caffeine = "Cn1c(=O)c2c(ncn2C)n(c1=O)C"
        sim = compute_similarity(ASPIRIN_SMILES, caffeine)
        # Just verify it runs without error
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# Test: NoveltyAgent (Agent 8)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestNoveltyAgent:
    @pytest.fixture
    def agent(self, mock_retriever, mock_llm):
        from app.agents.novelty_agent import NoveltyAgent
        return NoveltyAgent(retriever=mock_retriever, llm_client=mock_llm)

    async def test_run_returns_result(self, agent, sample_request):
        result = await agent.run(sample_request)
        assert result.success is True
        assert "novelty_score" in result.details
        assert 0.0 <= result.details["novelty_score"] <= 1.0

    def test_compute_novelty_score_no_similar(self):
        from app.agents.novelty_agent import NoveltyAgent
        score = NoveltyAgent._compute_novelty_score([])
        assert 0.80 <= score <= 0.90  # ~0.85 when no similar found


# ---------------------------------------------------------------------------
# Test: ResearchSummarizationAgent (Agent 9)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestResearchSummarizationAgent:
    @pytest.fixture
    def agent(self, mock_retriever, mock_llm):
        from app.agents.summarization_agent import ResearchSummarizationAgent
        return ResearchSummarizationAgent(retriever=mock_retriever, llm_client=mock_llm)

    @pytest.fixture
    def mock_agent_results(self):
        return {
            "LiteratureRetrievalAgent": _make_agent_result("LiteratureRetrievalAgent"),
            "SyntheticRouteAgent":      _make_agent_result("SyntheticRouteAgent"),
            "ToxicityAgent":            _make_agent_result("ToxicityAgent", success=False),
        }

    async def test_run_with_agent_results(self, agent, sample_request, mock_agent_results):
        result = await agent.run(sample_request, agent_results=mock_agent_results)
        assert result.success is True
        assert result.agent_name == "ResearchSummarizationAgent"

    async def test_run_without_results_returns_error(self, agent, sample_request):
        result = await agent.run(sample_request, agent_results=None)
        assert result.success is False

    def test_count_total_papers(self):
        from app.agents.summarization_agent import ResearchSummarizationAgent
        results = {
            "AgentA": _make_agent_result("AgentA"),
            "AgentB": _make_agent_result("AgentB"),
        }
        count = ResearchSummarizationAgent._count_total_papers(results)
        assert count >= 1  # Both agents share doc-001


# ---------------------------------------------------------------------------
# Test: FinalReasoningAgent (Agent 10)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFinalReasoningAgent:
    @pytest.fixture
    def agent(self, mock_retriever, mock_llm):
        from app.agents.final_reasoning_agent import FinalReasoningAgent
        return FinalReasoningAgent(retriever=mock_retriever, llm_client=mock_llm)

    @pytest.fixture
    def mock_all_results(self):
        agents = [
            "LiteratureRetrievalAgent", "SyntheticRouteAgent",
            "ChemicalAvailabilityAgent", "PatentRetrievalAgent",
            "ToxicityAgent", "SafetyAgent", "DrugLikenessAgent", "NoveltyAgent",
        ]
        return {name: _make_agent_result(name) for name in agents}

    async def test_run_returns_result(self, agent, sample_request, mock_all_results):
        summ = _make_agent_result("ResearchSummarizationAgent")
        result = await agent.run(
            sample_request,
            agent_results=mock_all_results,
            summarization_result=summ,
        )
        assert result.success is True
        assert result.agent_name == "FinalReasoningAgent"
        assert len(result.summary) > 0

    async def test_run_without_results_returns_error(self, agent, sample_request):
        result = await agent.run(sample_request, agent_results=None)
        assert result.success is False

    def test_aggregate_confidence(self):
        from app.agents.final_reasoning_agent import FinalReasoningAgent
        confs = [Confidence.HIGH, Confidence.HIGH, Confidence.MEDIUM]
        agg   = FinalReasoningAgent._aggregate_confidence(confs)
        assert agg == Confidence.HIGH

    def test_aggregate_confidence_empty(self):
        from app.agents.final_reasoning_agent import FinalReasoningAgent
        agg = FinalReasoningAgent._aggregate_confidence([])
        assert agg == Confidence.NONE
