"""tests/conftest.py – Shared fixtures for pytest-asyncio test suite."""

from __future__ import annotations

import os
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch

# Set test environment variables BEFORE importing the app
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("APP_ENV", "development")

from app.main import app


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def client():
    """
    Async HTTP test client that bypasses the Qdrant lifespan startup.
    Qdrant is mocked so tests run without a live vector database.
    """
    with patch("app.vectordb.qdrant_client.QdrantClientManager.connect", new_callable=AsyncMock):
        with patch("app.vectordb.qdrant_client.QdrantClientManager.disconnect", new_callable=AsyncMock):
            with patch(
                "app.vectordb.qdrant_client.QdrantClientManager.health_check",
                new_callable=AsyncMock,
                return_value={"status": "healthy", "collections": ["research_documents"]},
            ):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    yield ac


@pytest.fixture
def mock_qdrant_repository():
    """Return a MagicMock replacing QdrantRepository for unit tests."""
    with patch("app.vectordb.repository.QdrantRepository") as mock_cls:
        instance = MagicMock()
        instance.upsert_chunks = AsyncMock(return_value=5)
        instance.search_with_filter = AsyncMock(return_value=[])
        instance.search_with_vectors = AsyncMock(return_value=[])
        instance.delete_document_chunks = AsyncMock(return_value=10)
        instance.count_document_chunks = AsyncMock(return_value=10)
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def mock_embedding_provider():
    """Return a mock embedding provider that returns dummy vectors."""
    with patch("app.embeddings.embedding_service.get_embedding_provider") as mock_fn:
        provider = MagicMock()
        provider.model_name = "text-embedding-3-small"
        provider.dimensions = 1536
        provider.embed_texts = AsyncMock(return_value=[[0.1] * 1536])
        provider.embed_query = AsyncMock(return_value=[0.1] * 1536)
        mock_fn.return_value = provider
        yield provider


@pytest.fixture
def mock_llm_client():
    """Return a mock LLM client that returns a canned response."""
    from app.llm.openai_client import LLMResponse
    with patch("app.llm.openai_client.get_llm_client") as mock_fn:
        client = MagicMock()
        client.model_name = "gpt-4o"
        client.complete = AsyncMock(
            return_value=LLMResponse(
                content="This is a test answer from the RAG pipeline.",
                model="gpt-4o",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                finish_reason="stop",
            )
        )
        mock_fn.return_value = client
        yield client
