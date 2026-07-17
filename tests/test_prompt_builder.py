"""tests/test_prompt_builder.py – Unit tests for the PromptBuilder."""

from __future__ import annotations

import pytest

from app.prompts.prompt_builder import PromptBuilder, CONTEXT_HEADER
from app.schemas.chat import RetrievedChunk


def _make_chunk(
    chunk_id: str = "c1",
    doc_id: str = "d1",
    title: str = "Test Paper",
    page: int = 1,
    section: str = "Introduction",
    text: str = "This is chunk text about transformers.",
    score: float = 0.85,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        title=title,
        page=page,
        section=section,
        chunk_text=text,
        similarity_score=score,
    )


def test_build_returns_built_prompt():
    builder = PromptBuilder()
    chunk = _make_chunk()
    result = builder.build("What is a transformer?", [chunk])
    assert result.system_message
    assert result.user_message
    assert result.context_chunks_used == 1


def test_system_message_contains_sanjeevani():
    builder = PromptBuilder()
    result = builder.build("Test?", [_make_chunk()])
    assert "Sanjeevani" in result.system_message


def test_user_message_contains_question():
    builder = PromptBuilder()
    question = "What are the key findings?"
    result = builder.build(question, [_make_chunk()])
    assert question in result.user_message


def test_user_message_contains_context_header():
    builder = PromptBuilder()
    result = builder.build("Test?", [_make_chunk()])
    assert CONTEXT_HEADER in result.user_message


def test_user_message_contains_chunk_text():
    builder = PromptBuilder()
    chunk = _make_chunk(text="Unique chunk text about attention mechanisms.")
    result = builder.build("Test?", [chunk])
    assert "Unique chunk text about attention mechanisms." in result.user_message


def test_empty_chunks_returns_not_found_context():
    builder = PromptBuilder()
    result = builder.build("Test?", [])
    assert "No relevant context" in result.user_message
    assert result.context_chunks_used == 0


def test_deduplication_removes_identical_chunks():
    builder = PromptBuilder()
    # Two chunks with identical first 100 chars
    text = "A" * 200
    chunk1 = _make_chunk(chunk_id="c1", text=text, score=0.9)
    chunk2 = _make_chunk(chunk_id="c2", text=text, score=0.8)
    result = builder.build("Test?", [chunk1, chunk2])
    assert result.context_chunks_used == 1


def test_token_budget_respected():
    """Builder should not exceed max_context_tokens."""
    from app.utils.tokenizer import count_tokens
    builder = PromptBuilder(max_context_tokens=500)
    # Create 20 large chunks
    chunks = [_make_chunk(chunk_id=f"c{i}", text="word " * 100) for i in range(20)]
    result = builder.build("Test?", chunks)
    assert result.total_context_tokens <= 550  # Small buffer for formatting
