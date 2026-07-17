"""
Module 6: Embedding Service
app/embeddings/embedding_service.py

Provider-agnostic embedding service.
Currently supports OpenAI text-embedding-3-small.
Designed for easy addition of Gemini / local model providers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config.settings import EmbeddingProvider, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEmbeddingProvider(ABC):
    """Abstract contract for all embedding backends."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors (one per input).
        """

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """
        Generate a single embedding for a query string.

        Args:
            text: Query string.

        Returns:
            Single embedding vector.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding vector dimension."""


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI embedding provider using text-embedding-3-small (or configurable model).

    Features:
    - Async batch embedding with configurable batch size
    - Automatic retry with exponential back-off on rate-limit errors
    - Token-safe text truncation before embedding
    """

    # OpenAI allows up to 2048 inputs per request; keep conservative
    BATCH_SIZE = 100
    # text-embedding-3-small max input tokens
    MAX_INPUT_TOKENS = 8191

    def __init__(self) -> None:
        from openai import AsyncOpenAI
        from app.utils.tokenizer import truncate_to_tokens

        settings = get_settings()
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")

        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_request_timeout,
        )
        self._model = settings.openai_embedding_model
        self._dimensions = settings.openai_embedding_dimensions
        self._truncate = truncate_to_tokens
        logger.info("OpenAI embedding provider initialised with model '%s'", self._model)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching and retrying as needed."""
        if not texts:
            return []

        # Truncate each text to model token limit
        safe_texts = [
            self._truncate(t, self.MAX_INPUT_TOKENS) for t in texts
        ]

        all_embeddings: list[list[float]] = []

        for i in range(0, len(safe_texts), self.BATCH_SIZE):
            batch = safe_texts[i : i + self.BATCH_SIZE]
            batch_embeddings = await self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        results = await self.embed_texts([text])
        return results[0]

    async def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embeddings API with exponential back-off."""
        from openai import RateLimitError, APIStatusError

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((RateLimitError, APIStatusError)),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        ):
            with attempt:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                    encoding_format="float",
                )
                # Sort by index to guarantee order
                sorted_data = sorted(response.data, key=lambda d: d.index)
                return [item.embedding for item in sorted_data]

        raise RuntimeError("Embedding request failed after all retries")


# ---------------------------------------------------------------------------
# Gemini provider (future)
# ---------------------------------------------------------------------------

class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Placeholder for Google Gemini embedding provider.

    Will use the 'models/text-embedding-004' model when implemented.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "GeminiEmbeddingProvider is not yet implemented. "
            "Set EMBEDDING_PROVIDER=openai in your .env"
        )

    @property
    def model_name(self) -> str:
        return "models/text-embedding-004"

    @property
    def dimensions(self) -> int:
        return 768

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_embedding_provider() -> BaseEmbeddingProvider:
    """
    Instantiate the correct embedding provider based on EMBEDDING_PROVIDER setting.
    """
    settings = get_settings()
    provider = settings.embedding_provider

    if provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddingProvider()
    elif provider == EmbeddingProvider.GEMINI:
        return GeminiEmbeddingProvider()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------

_embedding_provider: BaseEmbeddingProvider | None = None


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Return the module-level embedding provider singleton."""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = create_embedding_provider()
    return _embedding_provider
