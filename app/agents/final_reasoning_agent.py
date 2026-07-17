"""
app/agents/final_reasoning_agent.py

Agent 10 – Final Reasoning Agent

Responsibilities:
- Receive outputs from ALL previous agents (1-9)
- Never discard evidence
- Resolve conflicting findings by explicitly stating disagreements
- Produce a structured report with citations and confidence levels
- Generate the final scientific review article narrative
"""

from __future__ import annotations

import logging
import time

from app.agents.base_agent import BaseAgent
from app.schemas.agents import AgentResult, Confidence
from app.schemas.analysis import AnalysisRequest

logger = logging.getLogger(__name__)


FINAL_REASONING_SYSTEM_PROMPT = """\
You are a Principal AI Research Analyst and Scientific Review Author.

You receive the complete outputs of a multi-agent RAG analysis system.
Your task is to produce a final, authoritative scientific research report
that reads like a peer-reviewed review article.

The report must:
1. Synthesize all agent findings into a coherent narrative.
2. Explicitly resolve conflicting findings by stating the disagreement and available evidence.
3. Never discard any evidence — all findings must be represented.
4. Clearly identify:
   - ESTABLISHED FINDINGS (high-confidence, multiple sources)
   - AREAS OF UNCERTAINTY (conflicting or limited evidence)
   - MISSING EVIDENCE (what gaps exist)
   - SUGGESTED FUTURE RESEARCH DIRECTIONS
5. Cite all evidence by [Agent] and [Source Document].
6. Include a structured confidence assessment for each section.

Report Structure:
EXECUTIVE SUMMARY:
[2-3 paragraph high-level summary]

MOLECULE OVERVIEW:
[Overview of the compound]

SYNTHESIS ASSESSMENT:
[Synthesis route findings with citations]

PATENT LANDSCAPE:
[Patent findings with numbers if found]

TOXICITY & SAFETY:
[Combined toxicity and safety assessment]

DRUG-LIKENESS:
[Descriptor-based assessment]

NOVELTY ASSESSMENT:
[Novelty findings with similarity data]

RESEARCH GAPS:
[Bullet list of gaps]

FUTURE RESEARCH DIRECTIONS:
[Suggested next steps]

CONFIDENCE ASSESSMENT:
[Per-section confidence: HIGH | MEDIUM | LOW]

REFERENCES:
[All cited sources]

Rules:
- Never fabricate citations, patent numbers, or toxicity values.
- Always distinguish: RETRIEVED EVIDENCE | COMPUTATIONAL PREDICTION | HEURISTIC
- Use scientific language appropriate for researchers.
- Every conclusion must reference its source.
"""


class FinalReasoningAgent(BaseAgent):
    """
    Agent 10: Final Reasoning Agent

    Receives all agent results (1-9) and produces the final structured report.
    This is the most comprehensive LLM call in the pipeline.
    """

    @property
    def agent_name(self) -> str:
        return "FinalReasoningAgent"

    async def run(
        self,
        request: AnalysisRequest,
        agent_results: dict[str, AgentResult] | None = None,
        summarization_result: AgentResult | None = None,
    ) -> AgentResult:
        start = time.perf_counter()
        try:
            if not agent_results:
                return self._error_result("No agent results provided", 0.0)

            primary_query = request.effective_query()

            # Build comprehensive context from all agent results + summarization
            context = self._build_final_context(
                agent_results, summarization_result
            )

            question = (
                f"Produce the final scientific review report for: '{primary_query}'. "
                f"Synthesize all agent findings into a structured, citation-rich review article. "
                f"Explicitly resolve any conflicting findings. "
                f"Assess confidence for each section. "
                f"Never discard any evidence."
            )

            prompt   = self._build_simple_prompt(
                FINAL_REASONING_SYSTEM_PROMPT,
                context,
                question,
            )
            llm_resp = await self._llm.complete(prompt)
            content  = llm_resp.content

            elapsed_ms = (time.perf_counter() - start) * 1000

            # Determine overall confidence
            agent_confidences = [r.confidence for r in agent_results.values() if r.success]
            overall_conf = self._aggregate_confidence(agent_confidences)

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                confidence=overall_conf,
                summary=content,
                evidence=[],    # Final reasoning synthesizes, doesn't retrieve
                details={
                    "final_report_narrative": content,
                    "model_used": llm_resp.model,
                    "prompt_tokens": llm_resp.prompt_tokens,
                    "completion_tokens": llm_resp.completion_tokens,
                    "total_tokens": llm_resp.total_tokens,
                },
                processing_time_ms=elapsed_ms,
                chunks_retrieved=0,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return self._error_result(exc, elapsed_ms)

    @staticmethod
    def _build_final_context(
        agent_results: dict[str, AgentResult],
        summarization_result: AgentResult | None,
    ) -> str:
        """Build the full context string for the final reasoning agent."""
        sections: list[str] = []

        if summarization_result and summarization_result.success:
            sections.append(
                f"=== RESEARCH SYNTHESIS (Agent 9) ===\n{summarization_result.summary[:2000]}"
            )

        for name, result in agent_results.items():
            if not result.success:
                sections.append(
                    f"=== {name} [FAILED] ===\n"
                    f"Error: {result.error or 'Unknown error'}"
                )
                continue

            conf_label = result.confidence.value.upper()
            # Truncate long summaries to keep within context window
            summary_text = result.summary[:1200] if result.summary else "No summary."

            # Add key details
            details_text = ""
            if result.details:
                important_keys = [
                    "routes_found", "patents_found", "availability_status",
                    "known_hazards", "novelty_score", "passes_lipinski",
                    "estimated_complexity", "molecular_weight", "qed",
                ]
                detail_lines = [
                    f"  {k}: {result.details[k]}"
                    for k in important_keys
                    if k in result.details and result.details[k] is not None
                ]
                if detail_lines:
                    details_text = "\nKey metrics:\n" + "\n".join(detail_lines)

            sections.append(
                f"=== {name} [Confidence: {conf_label}] ===\n"
                f"{summary_text}"
                f"{details_text}"
            )

        return "\n\n".join(sections)

    @staticmethod
    def _aggregate_confidence(confidences: list[Confidence]) -> Confidence:
        """Aggregate a list of confidence levels into one overall level."""
        if not confidences:
            return Confidence.NONE
        _order = {
            Confidence.HIGH: 3, Confidence.MEDIUM: 2,
            Confidence.LOW: 1, Confidence.NONE: 0,
        }
        avg = sum(_order[c] for c in confidences) / len(confidences)
        if avg >= 2.5:
            return Confidence.HIGH
        if avg >= 1.5:
            return Confidence.MEDIUM
        if avg >= 0.5:
            return Confidence.LOW
        return Confidence.NONE
