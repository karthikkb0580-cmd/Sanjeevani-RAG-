"""
Module 9: Prompt Builder
app/prompts/prompt_builder.py

Constructs the system + context + user prompt sent to the LLM.
The LLM MUST answer ONLY from retrieved context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.schemas.chat import RetrievedChunk
from app.utils.tokenizer import count_tokens, truncate_to_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Sanjeevani, an expert scientific research assistant with deep knowledge \
of academic literature. You answer questions STRICTLY based on the research \
documents provided in the context below.

Rules you MUST follow:
1. Answer ONLY using information present in the provided context.
2. If the context does not contain enough information to answer, respond with:
   "I could not find relevant information in the indexed documents."
3. Always cite the source document, page number, and section when you use \
   information from the context.
4. Do NOT fabricate, invent, or hallucinate any facts.
5. Be precise, concise, and scientifically accurate.
6. Use clear, professional language appropriate for researchers.
"""

CONTEXT_HEADER = "=== RETRIEVED CONTEXT FROM INDEXED DOCUMENTS ==="
CONTEXT_SEPARATOR = "---"
CONTEXT_FOOTER = "=== END OF CONTEXT ==="

INSTRUCTIONS = """\
Based ONLY on the context above, provide a comprehensive and accurate answer. \
Include specific citations in your response (e.g., "According to [Document Title], \
page X, section Y: ..."). If the answer cannot be found in the provided context, \
say so explicitly.
"""

# Max tokens to allocate for context (leaving room for system + question + response)
MAX_CONTEXT_TOKENS = 6000


@dataclass
class BuiltPrompt:
    """Container for the structured prompt sent to the LLM."""
    system_message: str
    user_message: str
    context_chunks_used: int
    total_context_tokens: int


class PromptBuilder:
    """
    Builds structured prompts for the LLM.

    Structure:
        SYSTEM
        ↓
        Retrieved Context (deduplicated, token-capped)
        ↓
        User Question
        ↓
        Instructions
    """

    def __init__(self, max_context_tokens: int = MAX_CONTEXT_TOKENS) -> None:
        self.max_context_tokens = max_context_tokens

    def build(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> BuiltPrompt:
        """
        Build the complete prompt from a question and retrieved chunks.

        Args:
            question: The user's question.
            chunks: Re-ranked retrieved chunks.

        Returns:
            BuiltPrompt with system message, user message, and token stats.
        """
        context_str, chunks_used, context_tokens = self._build_context(chunks)
        user_message = self._build_user_message(question, context_str)

        logger.debug(
            "Prompt built: %d chunks, %d context tokens, %d user msg tokens",
            chunks_used,
            context_tokens,
            count_tokens(user_message),
        )

        return BuiltPrompt(
            system_message=SYSTEM_PROMPT,
            user_message=user_message,
            context_chunks_used=chunks_used,
            total_context_tokens=context_tokens,
        )

    def _build_context(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[str, int, int]:
        """
        Assemble deduplicated context from chunks, respecting token budget.

        Returns:
            (context_string, chunks_included, total_tokens_used)
        """
        if not chunks:
            return (
                "No relevant context was found in the indexed documents.",
                0,
                count_tokens("No relevant context was found in the indexed documents."),
            )

        context_parts: list[str] = []
        total_tokens = 0
        chunks_used = 0
        seen_texts: set[str] = set()

        for i, chunk in enumerate(chunks, start=1):
            # Simple deduplication: skip chunks with identical first 100 chars
            fingerprint = chunk.chunk_text[:100].strip()
            if fingerprint in seen_texts:
                continue
            seen_texts.add(fingerprint)

            chunk_block = self._format_chunk(i, chunk)
            chunk_tokens = count_tokens(chunk_block)

            if total_tokens + chunk_tokens > self.max_context_tokens:
                # Try truncating to fit the remaining budget
                remaining = self.max_context_tokens - total_tokens
                if remaining < 100:
                    break
                chunk_block = truncate_to_tokens(chunk_block, remaining)
                chunk_tokens = count_tokens(chunk_block)

            context_parts.append(chunk_block)
            total_tokens += chunk_tokens
            chunks_used += 1

        context_str = f"\n{CONTEXT_SEPARATOR}\n".join(context_parts)
        return context_str, chunks_used, total_tokens

    @staticmethod
    def _format_chunk(index: int, chunk: RetrievedChunk) -> str:
        """Format a single chunk into a readable context block."""
        lines = [
            f"[{index}] Document: {chunk.title}",
            f"    Page: {chunk.page} | Section: {chunk.section or 'N/A'}",
            f"    Similarity: {chunk.similarity_score:.3f}",
            "",
            chunk.chunk_text,
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_user_message(question: str, context: str) -> str:
        """Assemble the full user message from context and question."""
        return (
            f"{CONTEXT_HEADER}\n\n"
            f"{context}\n\n"
            f"{CONTEXT_FOOTER}\n\n"
            f"QUESTION: {question}\n\n"
            f"{INSTRUCTIONS}"
        )
