"""tests/test_gemini_providers.py – Unit tests for Gemini providers."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from app.config.settings import get_settings, EmbeddingProvider, LLMProvider
from app.embeddings.embedding_service import GeminiEmbeddingProvider
from app.llm.openai_client import GeminiLLMClient
from app.prompts.prompt_builder import BuiltPrompt


@pytest.mark.asyncio
async def test_gemini_embedding_provider_success():
    # Mock settings
    settings = get_settings()
    settings.gemini_api_key = "fake-key"
    settings.embedding_provider = EmbeddingProvider.GEMINI

    # Mock response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={
        "embeddings": [
            {"values": [0.1, 0.2, 0.3]},
            {"values": [0.4, 0.5, 0.6]}
        ]
    })

    # Patch AsyncClient.post
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        provider = GeminiEmbeddingProvider()
        assert provider.model_name == "models/gemini-embedding-001"
        assert provider.dimensions == 768

        # Test embed_query
        vector = await provider.embed_query("hello")
        assert vector == [0.1, 0.2, 0.3]

        # Test embed_texts
        vectors = await provider.embed_texts(["hello", "world"])
        assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.mark.asyncio
async def test_gemini_embedding_provider_error():
    # Mock settings
    settings = get_settings()
    settings.gemini_api_key = "fake-key"

    # Mock response with error
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.text = "Invalid Key"
    mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
        message="Bad Request",
        request=MagicMock(),
        response=mock_response
    ))

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        provider = GeminiEmbeddingProvider()
        with pytest.raises(httpx.HTTPStatusError):
            await provider.embed_query("hello")


@pytest.mark.asyncio
async def test_gemini_llm_client_success():
    # Mock settings
    settings = get_settings()
    settings.gemini_api_key = "fake-key"
    settings.llm_provider = LLMProvider.GEMINI

    # Mock response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello, I am Gemini!"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15
        }
    })

    # Patch AsyncClient.post
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        client = GeminiLLMClient()
        assert client.model_name == "models/gemini-3.5-flash"

        prompt = BuiltPrompt(
            system_message="system prompt",
            user_message="user prompt",
            context_chunks_used=[],
            total_context_tokens=0
        )
        response = await client.complete(prompt)
        assert response.content == "Hello, I am Gemini!"
        assert response.model == "models/gemini-3.5-flash"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5
        assert response.total_tokens == 15
        assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_gemini_llm_client_error():
    # Mock settings
    settings = get_settings()
    settings.gemini_api_key = "fake-key"

    # Mock response with error
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
        message="Server Error",
        request=MagicMock(),
        response=mock_response
    ))

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        client = GeminiLLMClient()
        prompt = BuiltPrompt(
            system_message="sys",
            user_message="user",
            context_chunks_used=[],
            total_context_tokens=0
        )
        with pytest.raises(httpx.HTTPStatusError):
            await client.complete(prompt)
