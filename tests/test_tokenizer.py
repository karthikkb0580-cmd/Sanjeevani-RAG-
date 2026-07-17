"""tests/test_tokenizer.py – Unit tests for the tokenizer utility."""

from __future__ import annotations

import pytest

from app.utils.tokenizer import count_tokens, truncate_to_tokens


def test_count_tokens_empty_string():
    assert count_tokens("") == 0


def test_count_tokens_single_word():
    count = count_tokens("hello")
    assert count >= 1


def test_count_tokens_known_phrase():
    # "Hello world" should be 2 tokens in cl100k_base
    count = count_tokens("Hello world")
    assert 1 <= count <= 3


def test_count_tokens_long_text():
    text = "The quick brown fox jumps over the lazy dog. " * 100
    count = count_tokens(text)
    assert count > 100


def test_truncate_within_limit():
    text = "hello world"
    result = truncate_to_tokens(text, max_tokens=100)
    assert result == text


def test_truncate_to_zero():
    text = "This is a test sentence."
    result = truncate_to_tokens(text, max_tokens=0)
    assert result == ""


def test_truncate_reduces_length():
    text = "word " * 1000
    max_tokens = 50
    result = truncate_to_tokens(text, max_tokens=max_tokens)
    assert count_tokens(result) <= max_tokens


def test_truncate_preserves_text_when_short():
    short = "Hi."
    result = truncate_to_tokens(short, max_tokens=100)
    assert result == short
