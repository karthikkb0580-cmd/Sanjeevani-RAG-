"""
app/api/analysis.py

Multi-Agent Analysis API Router
POST /api/v2/analyze

Provides:
  - Standard JSON response
  - SSE streaming response
  - Background task support
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.orchestrator.orchestrator import AnalysisOrchestrator, get_orchestrator
from app.schemas.analysis import AnalysisRequest, FinalAnalysisReport

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Multi-Agent Analysis"])


# ---------------------------------------------------------------------------
# Dependency: shared orchestrator
# ---------------------------------------------------------------------------

def get_analysis_orchestrator() -> AnalysisOrchestrator:
    """FastAPI dependency that returns the singleton orchestrator."""
    return get_orchestrator()


# ---------------------------------------------------------------------------
# POST /api/v2/analyze
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=FinalAnalysisReport,
    summary="Run Multi-Agent RAG Analysis",
    description=(
        "Orchestrates 10 specialized AI agents in parallel to analyze a molecule, "
        "research query, or SMILES string. Returns a structured scientific report "
        "with literature evidence, synthesis routes, patents, toxicity, safety, "
        "drug-likeness, novelty, and confidence assessments. "
        "Set `stream=true` to receive a Server-Sent Events (SSE) stream."
    ),
    response_description="Structured multi-agent analysis report",
    status_code=status.HTTP_200_OK,
)
async def run_analysis(
    request: AnalysisRequest,
    orchestrator: AnalysisOrchestrator = Depends(get_analysis_orchestrator),
) -> FinalAnalysisReport | StreamingResponse:
    """
    Main multi-agent analysis endpoint.

    Accepts either a free-text query, a structured molecule input, or both.
    Runs all 10 analysis agents and returns a comprehensive scientific report.
    """
    # Validate that at least one input is provided
    try:
        query_text = request.effective_query()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    logger.info(
        "Analysis request received: query='%s…' stream=%s",
        query_text[:60],
        request.stream,
    )

    if request.stream:
        # Return SSE stream
        return StreamingResponse(
            orchestrator.stream_analyze(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control":    "no-cache",
                "Connection":       "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Standard JSON response
    try:
        report = await orchestrator.analyze(request)
        logger.info(
            "Analysis complete: %d agents succeeded, %d failed, %.0f ms",
            report.agents_succeeded,
            report.agents_failed,
            report.total_processing_time_ms,
        )
        return report
    except Exception as exc:
        logger.exception("Analysis pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# GET /api/v2/agents
# ---------------------------------------------------------------------------

@router.get(
    "/agents",
    summary="List Available Agents",
    description="Returns the list of all registered analysis agents.",
)
async def list_agents() -> dict:
    """Return metadata about all registered agents."""
    from app.agents import PARALLEL_AGENTS

    return {
        "parallel_agents": [cls.__name__ for cls in PARALLEL_AGENTS],
        "sequential_agents": [
            "ResearchSummarizationAgent",
            "FinalReasoningAgent",
        ],
        "total_agents": len(PARALLEL_AGENTS) + 2,
        "execution_model": "Agents 1-8 run in parallel via asyncio.gather(). "
                           "Agent 9 (Summarization) runs after. "
                           "Agent 10 (Final Reasoning) runs last.",
    }


# ---------------------------------------------------------------------------
# POST /api/v2/analyze/agent/{agent_name}
# ---------------------------------------------------------------------------

@router.post(
    "/analyze/agent/{agent_name}",
    summary="Run a Single Agent",
    description=(
        "Run a single analysis agent in isolation. "
        "Useful for testing individual agents without running the full pipeline."
    ),
)
async def run_single_agent(
    agent_name: str,
    request: AnalysisRequest,
) -> dict:
    """
    Run a single named agent independently.

    Useful for debugging, testing, or building custom pipelines.
    """
    from app.agents import PARALLEL_AGENTS
    from app.llm.openai_client import get_llm_client
    from app.retrieval.retriever import Retriever

    # Find the agent class
    agent_class = next(
        (cls for cls in PARALLEL_AGENTS if cls.__name__ == agent_name),
        None,
    )

    if not agent_class:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Agent '{agent_name}' not found. "
                f"Available: {[cls.__name__ for cls in PARALLEL_AGENTS]}"
            ),
        )

    try:
        query_text = request.effective_query()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # Instantiate and run
    try:
        agent    = agent_class(retriever=Retriever(), llm_client=get_llm_client())
        result   = await agent.run(request)
        return result.model_dump(mode="json")
    except Exception as exc:
        logger.exception("Single agent run failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution error: {str(exc)}",
        )
