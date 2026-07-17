"""tests/test_chat_service.py – Unit tests for ChatService agentic RAG chat loop."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.chat import ChatRequest, RetrievedChunk
from app.services.chat_service import ChatService
from app.llm.openai_client import LLMResponse


@pytest.mark.asyncio
async def test_chat_service_agentic_loop_answer_directly():
    # Setup mocks
    mock_chunk = RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-123",
        title="Test Document",
        page=1,
        section="Intro",
        chunk_text="This is relevant information.",
        similarity_score=0.85
    )

    with patch("app.services.chat_service.Retriever") as mock_retriever_cls, \
         patch("app.services.chat_service.Reranker") as mock_reranker_cls, \
         patch("app.services.chat_service.PromptBuilder") as mock_prompt_builder_cls, \
         patch("app.services.chat_service.get_llm_client") as mock_get_llm_client, \
         patch("app.services.chat_service.get_embedding_provider") as mock_get_embed_provider:

        # Mock Retriever
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve = AsyncMock(return_value=[mock_chunk])

        # Mock Reranker
        mock_reranker = mock_reranker_cls.return_value
        mock_reranker.rerank = MagicMock(return_value=[mock_chunk])

        # Mock LLM Client
        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"
        
        # We need two LLM responses:
        # 1. Routing decision: {"action": "answer", "reasoning": "We have enough info."}
        # 2. Final answer: "This is the final answer."
        mock_llm.complete = AsyncMock(side_effect=[
            LLMResponse(
                content='{"action": "answer", "reasoning": "We have enough info.", "query": ""}',
                model="test-model",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                finish_reason="stop"
            ),
            LLMResponse(
                content="This is the final answer.",
                model="test-model",
                prompt_tokens=100,
                completion_tokens=30,
                total_tokens=130,
                finish_reason="stop"
            )
        ])
        mock_get_llm_client.return_value = mock_llm

        # Mock Embedder
        mock_embedder = MagicMock()
        mock_embedder.model_name = "test-embedding-model"
        mock_get_embed_provider.return_value = mock_embedder

        service = ChatService()
        request = ChatRequest(question="What is the answer?", top_k=3, similarity_threshold=0.6)
        
        response = await service.chat(request)
        
        assert response.question == "What is the answer?"
        assert response.answer == "This is the final answer."
        assert len(response.citations) == 1
        assert response.total_chunks_retrieved == 1
        assert mock_retriever.retrieve.call_count == 1


@pytest.mark.asyncio
async def test_chat_service_agentic_loop_search_then_answer():
    # Setup mocks
    mock_chunk_1 = RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-123",
        title="Test Document",
        page=1,
        section="Intro",
        chunk_text="This is initial info.",
        similarity_score=0.85
    )
    mock_chunk_2 = RetrievedChunk(
        chunk_id="chunk-2",
        document_id="doc-123",
        title="Test Document",
        page=2,
        section="Body",
        chunk_text="This is detailed info.",
        similarity_score=0.90
    )

    with patch("app.services.chat_service.Retriever") as mock_retriever_cls, \
         patch("app.services.chat_service.Reranker") as mock_reranker_cls, \
         patch("app.services.chat_service.PromptBuilder") as mock_prompt_builder_cls, \
         patch("app.services.chat_service.get_llm_client") as mock_get_llm_client, \
         patch("app.services.chat_service.get_embedding_provider") as mock_get_embed_provider:

        # Mock Retriever
        mock_retriever = mock_retriever_cls.return_value
        # First retrieve returns mock_chunk_1, second retrieve returns mock_chunk_2
        mock_retriever.retrieve = AsyncMock(side_effect=[[mock_chunk_1], [mock_chunk_2]])

        # Mock Reranker
        mock_reranker = mock_reranker_cls.return_value
        mock_reranker.rerank = MagicMock(return_value=[mock_chunk_1, mock_chunk_2])

        # Mock LLM Client
        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"
        
        # We need three LLM responses:
        # 1. Routing decision 1: {"action": "search", "reasoning": "Need more details", "query": "more details"}
        # 2. Routing decision 2: {"action": "answer", "reasoning": "We have enough info."}
        # 3. Final answer: "This is the final answer."
        mock_llm.complete = AsyncMock(side_effect=[
            LLMResponse(
                content='{"action": "search", "reasoning": "Need more details", "query": "more details"}',
                model="test-model",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                finish_reason="stop"
            ),
            LLMResponse(
                content='{"action": "answer", "reasoning": "We have enough info.", "query": ""}',
                model="test-model",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                finish_reason="stop"
            ),
            LLMResponse(
                content="This is the final answer.",
                model="test-model",
                prompt_tokens=100,
                completion_tokens=30,
                total_tokens=130,
                finish_reason="stop"
            )
        ])
        mock_get_llm_client.return_value = mock_llm

        # Mock Embedder
        mock_embedder = MagicMock()
        mock_embedder.model_name = "test-embedding-model"
        mock_get_embed_provider.return_value = mock_embedder

        service = ChatService()
        request = ChatRequest(question="What is the answer?", top_k=3, similarity_threshold=0.6)
        
        response = await service.chat(request)
        
        assert response.question == "What is the answer?"
        assert response.answer == "This is the final answer."
        assert response.total_chunks_retrieved == 2
        assert mock_retriever.retrieve.call_count == 2


@pytest.mark.asyncio
async def test_chat_service_fallback_when_no_chunks():
    with patch("app.services.chat_service.Retriever") as mock_retriever_cls, \
         patch("app.services.chat_service.Reranker") as mock_reranker_cls, \
         patch("app.services.chat_service.PromptBuilder") as mock_prompt_builder_cls, \
         patch("app.services.chat_service.get_llm_client") as mock_get_llm_client, \
         patch("app.services.chat_service.get_embedding_provider") as mock_get_embed_provider:

        # Mock Retriever to return empty list
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve = AsyncMock(return_value=[])

        # Mock Reranker
        mock_reranker = mock_reranker_cls.return_value
        mock_reranker.rerank = MagicMock(return_value=[])

        # Mock LLM Client
        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"
        mock_llm.complete = AsyncMock(return_value=LLMResponse(
            content="No relevant documents were found in the indexed knowledge base. The following response is based on general scientific knowledge: This is general knowledge.",
            model="test-model",
            prompt_tokens=40,
            completion_tokens=25,
            total_tokens=65,
            finish_reason="stop"
        ))
        mock_get_llm_client.return_value = mock_llm

        # Mock Embedder
        mock_embedder = MagicMock()
        mock_embedder.model_name = "test-embedding-model"
        mock_get_embed_provider.return_value = mock_embedder

        service = ChatService()
        request = ChatRequest(question="What is Ibuprofen?", top_k=3, similarity_threshold=0.6)
        
        response = await service.chat(request)
        
        assert response.question == "What is Ibuprofen?"
        assert "No relevant documents were found in the indexed knowledge base" in response.answer
        assert response.total_chunks_retrieved == 0
        assert mock_retriever.retrieve.call_count == 1

