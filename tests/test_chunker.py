"""tests/test_chunker.py – Unit tests for the DocumentChunker."""

from __future__ import annotations

import pytest

from app.chunking.chunker import DocumentChunker


SAMPLE_DOCUMENT = {
    "title": "Test Research Paper",
    "author": "Test Author",
    "file_path": "/tmp/test.pdf",
    "total_pages": 2,
    "metadata": {},
    "pages": [
        {
            "page": 1,
            "text": (
                "[SECTION: Introduction]\n\n"
                "This is the introduction section of the research paper. "
                "It contains several sentences that describe the motivation "
                "and objectives of the study. We investigate the properties "
                "of transformer-based language models and their applications "
                "in retrieval-augmented generation systems.\n\n"
                "The remainder of this paper is organised as follows. "
                "Section 2 reviews related work. Section 3 describes the "
                "experimental setup. Section 4 presents results."
            ),
            "headings": ["Introduction"],
            "section": "Introduction",
        },
        {
            "page": 2,
            "text": (
                "[SECTION: Methodology]\n\n"
                "We propose a novel retrieval pipeline that combines semantic "
                "similarity search with maximal marginal relevance reranking. "
                "Documents are first chunked using a recursive character splitter "
                "with a token-based size of 600 tokens and an overlap of 100 tokens. "
                "Each chunk is embedded using OpenAI text-embedding-3-small and "
                "stored in a Qdrant vector database for efficient retrieval."
            ),
            "headings": ["Methodology"],
            "section": "Methodology",
        },
    ],
}

SHORT_DOCUMENT = {
    "title": "Short Doc",
    "total_pages": 1,
    "metadata": {},
    "pages": [
        {"page": 1, "text": "Hi.", "headings": [], "section": ""},
    ],
}


def test_chunk_document_returns_list():
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(SAMPLE_DOCUMENT, "doc-001")
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunks_have_correct_document_id():
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(SAMPLE_DOCUMENT, "doc-xyz")
    for chunk in chunks:
        assert chunk.document_id == "doc-xyz"


def test_chunks_have_text():
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(SAMPLE_DOCUMENT, "doc-001")
    for chunk in chunks:
        assert chunk.text.strip() != ""


def test_chunks_have_page_numbers():
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(SAMPLE_DOCUMENT, "doc-001")
    pages = {chunk.page for chunk in chunks}
    assert 1 in pages
    assert 2 in pages


def test_chunks_have_sequential_indices():
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(SAMPLE_DOCUMENT, "doc-001")
    indices = [chunk.chunk_index for chunk in chunks]
    assert indices == list(range(len(chunks)))


def test_chunks_total_count_consistent():
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(SAMPLE_DOCUMENT, "doc-001")
    total = len(chunks)
    for chunk in chunks:
        assert chunk.total_chunks == total


def test_short_document_returns_empty_or_single():
    """A very short document below min_size should produce 0 or 1 chunk."""
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50, chunk_min_size=50)
    chunks = chunker.chunk_document(SHORT_DOCUMENT, "doc-short")
    assert len(chunks) <= 1


def test_chunk_ids_are_unique():
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(SAMPLE_DOCUMENT, "doc-001")
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_chunks_have_title():
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(SAMPLE_DOCUMENT, "doc-001")
    for chunk in chunks:
        assert chunk.title == "Test Research Paper"
