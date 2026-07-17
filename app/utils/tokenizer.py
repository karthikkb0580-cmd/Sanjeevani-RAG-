"""
Module utilities: Tokenizer
app/utils/tokenizer.py

Token counting utility using tiktoken.
Used by the chunker and prompt builder.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import tiktoken

logger = logging.getLogger(__name__)

# Default encoding used by OpenAI text-embedding-3-small and GPT-4+
_DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=4)
def _get_encoding(encoding_name: str) -> tiktoken.Encoding:
    """Return a cached tiktoken encoding."""
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = _DEFAULT_ENCODING) -> int:
    """
    Count the number of tokens in a string.

    Args:
        text: Input text.
        encoding_name: tiktoken encoding name (default: cl100k_base).

    Returns:
        Integer token count.
    """
    enc = _get_encoding(encoding_name)
    return len(enc.encode(text, disallowed_special=()))


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    encoding_name: str = _DEFAULT_ENCODING,
) -> str:
    """
    Truncate text to at most max_tokens tokens, decoding back to a string.

    Args:
        text: Input text.
        max_tokens: Maximum token count.
        encoding_name: tiktoken encoding name.

    Returns:
        Truncated string.
    """
    enc = _get_encoding(encoding_name)
    token_ids = enc.encode(text, disallowed_special=())
    if len(token_ids) <= max_tokens:
        return text
    return enc.decode(token_ids[:max_tokens])
