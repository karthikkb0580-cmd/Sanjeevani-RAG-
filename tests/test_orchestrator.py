"""
tests/test_orchestrator.py

Integration tests for the AnalysisOrchestrator.

Tests:
- Full pipeline with fully mocked agents
- Partial failures (some agents fail) don't abort the pipeline
- SSE streaming generates expected events
- Single-agent endpoint
- ConfidenceAssessment aggregation
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.agents import AgentResult, Confidence, EvidenceItem
from app.schemas.analysis import AnalysisRequest, MoleculeInput, FinalAnalysisReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture
def full_request() -> AnalysisRequest:
    return AnalysisRequest(
        query="Aspirin anti-inflammatory mechanism",
        molecule=MoleculeInput(
            name="Aspirin",
            smiles=ASPIRIN_SMILES,
            cas="50-78-2",
        ),
        top_k=5,
        similarity_threshold=0.60,
        stream=False,
    )


def _mock_agent_result(name: str, success: bool = True) -> AgentResult:
    return AgentResult(
        agent_name=name,
        success=success,
        confidence=Confidence.MEDIUM,
        summary=f"Mocked summary for {name}.",
        evidence=[
            EvidenceItem(
                document_id=f"doc-{name}",
                title=f"Test Paper ({name})",
                page=1,
                section="Methods",
                chunk_text="Test evidence.",
                similarity_score=0.80,
            )
        ],
        chunks_retrieved=3,
        error=None if success else f"{name} simulated error",
    )


# ---------------------------------------------------------------------------
# Mock orchestrator factory
# ---------------------------------------------------------------------------

def _make_mock_orchestrator():
    """
    Create a fully mocked AnalysisOrchestrator where all agents
    return pre-defined AgentResult objects.
    """
    from app.agents import PARALLEL_AGENTS
    from app.orchestrator.orchestrator import AnalysisOrchestrator

    with patch.object(AnalysisOrchestrator, "__init__", lambda self: None):
        orchestrator = AnalysisOrchestrator.__new__(AnalysisOrchestrator)

    # Mock LLM
    from app.llm.openai_client import LLMResponse
    mock_llm = AsyncMock()
    mock_llm.model_name = "mock-gpt-4o"
    mock_llm.complete.return_value = LLMResponse(
        content="Mock final report narrative with comprehensive analysis.",
        model="mock-gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        finish_reason="stop",
    )

    # Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.model_name = "mock-embedding-3-small"

    orchestrator._llm      = mock_llm
    orchestrator._embedder = mock_embedder

    # Mock parallel agents
    mock_agents = []
    for cls in PARALLEL_AGENTS:
        mock_agent = AsyncMock()
        mock_agent.agent_name = cls.__name__
        mock_agent.run.return_value = _mock_agent_result(cls.__name__)
        mock_agents.append(mock_agent)
    orchestrator._parallel_agents = mock_agents

    # Mock summarization agent
    summ_agent = AsyncMock()
    summ_agent.agent_name = "ResearchSummarizationAgent"
    summ_agent.run.return_value = _mock_agent_result("ResearchSummarizationAgent")
    summ_agent.run.return_value.details = {
        "key_findings":     ["Finding 1", "Finding 2"],
        "limitations":      ["Limited data"],
        "consensus_points": ["Aspirin inhibits COX"],
        "research_gaps":    ["Gap 1", "Gap 2"],
        "synthesis_narrative": "Comprehensive synthesis of findings.",
        "paper_count":      5,
    }
    orchestrator._summarization_agent = summ_agent

    # Mock final agent
    final_agent = AsyncMock()
    final_agent.agent_name = "FinalReasoningAgent"
    final_agent.run.return_value = _mock_agent_result("FinalReasoningAgent")
    final_agent.run.return_value.details = {
        "final_report_narrative": "Mock final report.",
        "model_used": "mock-gpt-4o",
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
    }
    orchestrator._final_agent = final_agent

    return orchestrator


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAnalysisOrchestrator:

    async def test_full_pipeline_returns_final_report(self, full_request):
        """Full pipeline should return a FinalAnalysisReport."""
        orchestrator = _make_mock_orchestrator()
        report = await orchestrator.analyze(full_request)
        assert isinstance(report, FinalAnalysisReport)

    async def test_report_has_request_id(self, full_request):
        orchestrator = _make_mock_orchestrator()
        report = await orchestrator.analyze(full_request)
        assert report.request_id
        assert len(report.request_id) == 36  # UUID4

    async def test_report_has_correct_query(self, full_request):
        orchestrator = _make_mock_orchestrator()
        report = await orchestrator.analyze(full_request)
        assert "Aspirin" in report.query

    async def test_all_8_parallel_agents_ran(self, full_request):
        from app.agents import PARALLEL_AGENTS
        orchestrator = _make_mock_orchestrator()
        report = await orchestrator.analyze(full_request)
        assert report.agents_succeeded + report.agents_failed == len(PARALLEL_AGENTS)

    async def test_partial_agent_failure_does_not_abort_pipeline(self, full_request):
        """If 2 agents fail, the pipeline should still complete."""
        orchestrator = _make_mock_orchestrator()
        # Make 2 agents return failure results
        orchestrator._parallel_agents[0].run.return_value = _mock_agent_result(
            "LiteratureRetrievalAgent", success=False
        )
        orchestrator._parallel_agents[1].run.return_value = _mock_agent_result(
            "SyntheticRouteAgent", success=False
        )
        report = await orchestrator.analyze(full_request)
        # Pipeline should complete
        assert isinstance(report, FinalAnalysisReport)
        assert report.agents_failed == 2
        assert report.agents_succeeded >= 6

    async def test_report_contains_research_gaps(self, full_request):
        orchestrator = _make_mock_orchestrator()
        report = await orchestrator.analyze(full_request)
        assert isinstance(report.research_gaps, list)

    async def test_report_has_confidence_assessment(self, full_request):
        orchestrator = _make_mock_orchestrator()
        report = await orchestrator.analyze(full_request)
        assert report.confidence_assessment is not None
        assert report.confidence_assessment.overall in Confidence.__members__.values()

    async def test_report_references_deduplicated(self, full_request):
        orchestrator = _make_mock_orchestrator()
        report = await orchestrator.analyze(full_request)
        # References should be unique
        assert len(report.references) == len(set(report.references))

    async def test_processing_time_recorded(self, full_request):
        orchestrator = _make_mock_orchestrator()
        report = await orchestrator.analyze(full_request)
        assert report.total_processing_time_ms > 0

    async def test_filter_agents_when_agents_enabled(self, full_request):
        orchestrator = _make_mock_orchestrator()
        full_request.agents_enabled = ["LiteratureRetrievalAgent"]
        active = orchestrator._filter_agents(full_request)
        assert len(active) == 1
        assert active[0].agent_name == "LiteratureRetrievalAgent"

    async def test_filter_agents_none_means_all(self, full_request):
        from app.agents import PARALLEL_AGENTS
        orchestrator = _make_mock_orchestrator()
        full_request.agents_enabled = None
        active = orchestrator._filter_agents(full_request)
        assert len(active) == len(PARALLEL_AGENTS)


# ---------------------------------------------------------------------------
# Test: SSE streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestStreamingOrchestrator:

    async def test_stream_yields_sse_events(self, full_request):
        """Streaming should yield start, phase, agent_complete, and complete events."""
        import json

        full_request.stream = True
        orchestrator = _make_mock_orchestrator()

        events_seen: list[str] = []
        async for raw_event in orchestrator.stream_analyze(full_request):
            if raw_event.startswith("event:"):
                event_name = raw_event.split("\n")[0].replace("event: ", "").strip()
                events_seen.append(event_name)

        assert "start"    in events_seen
        assert "complete" in events_seen
        assert "report"   in events_seen
        assert "phase"    in events_seen

    async def test_stream_complete_event_has_request_id(self, full_request):
        import json

        full_request.stream = True
        orchestrator = _make_mock_orchestrator()

        async for raw_event in orchestrator.stream_analyze(full_request):
            if "event: complete" in raw_event:
                data_line = [l for l in raw_event.split("\n") if l.startswith("data:")]
                if data_line:
                    data = json.loads(data_line[0].replace("data: ", ""))
                    assert "request_id" in data
                break


# ---------------------------------------------------------------------------
# Test: ConfidenceAssessment
# ---------------------------------------------------------------------------

class TestConfidenceAssessment:
    def test_overall_from_agents_high(self):
        from app.schemas.analysis import ConfidenceAssessment

        results = {
            name: _mock_agent_result(name)
            for name in [
                "LiteratureRetrievalAgent", "SyntheticRouteAgent",
                "ChemicalAvailabilityAgent", "PatentRetrievalAgent",
                "ToxicityAgent", "SafetyAgent",
            ]
        }
        # All MEDIUM → overall MEDIUM
        conf = ConfidenceAssessment()
        conf.overall_from_agents(results)
        assert conf.overall in (Confidence.MEDIUM, Confidence.HIGH)

    def test_overall_from_agents_empty(self):
        from app.schemas.analysis import ConfidenceAssessment
        conf = ConfidenceAssessment()
        conf.overall_from_agents({})
        assert conf.overall == Confidence.NONE


# ---------------------------------------------------------------------------
# Test: Build references
# ---------------------------------------------------------------------------

class TestBuildReferences:
    def test_deduplicated_references(self):
        from app.orchestrator.orchestrator import AnalysisOrchestrator

        # Two agents with the same evidence → should not duplicate
        agent_a = _mock_agent_result("AgentA")
        agent_b = _mock_agent_result("AgentA")  # same evidence
        results = {"AgentA": agent_a, "AgentB": agent_b}
        refs = AnalysisOrchestrator._build_references(results)
        assert len(refs) == len(set(refs))


# ---------------------------------------------------------------------------
# Test: API endpoint (smoke test)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAnalysisAPIEndpoint:
    async def test_analyze_endpoint_structure(self, full_request):
        """
        Smoke test: verify the analyze endpoint returns expected keys.
        (Mocked orchestrator – no real LLM/Qdrant calls)
        """
        from app.api.analysis import run_analysis

        orchestrator = _make_mock_orchestrator()

        # Call the endpoint function directly with mocked orchestrator
        result = await run_analysis(request=full_request, orchestrator=orchestrator)
        assert isinstance(result, FinalAnalysisReport)
        assert result.request_id
        assert result.confidence_assessment is not None

    async def test_analyze_endpoint_raises_422_without_input(self):
        """Endpoint should raise 422 when neither query nor molecule is provided."""
        from fastapi import HTTPException
        from app.api.analysis import run_analysis

        empty_request = AnalysisRequest()
        orchestrator  = _make_mock_orchestrator()

        with pytest.raises(HTTPException) as exc_info:
            await run_analysis(request=empty_request, orchestrator=orchestrator)
        assert exc_info.value.status_code == 422
