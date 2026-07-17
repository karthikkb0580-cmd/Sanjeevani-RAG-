"""
Module 10: OpenAI LLM Client
app/llm/openai_client.py

Async OpenAI chat completion client with retry logic.
Supports future Gemini integration via the same interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config.settings import LLMProvider, get_settings
from app.prompts.prompt_builder import BuiltPrompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Structured LLM completion response."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """Abstract LLM client contract."""

    @abstractmethod
    async def complete(self, prompt: BuiltPrompt) -> LLMResponse:
        """Send a BuiltPrompt to the LLM and return the response."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model identifier."""


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAILLMClient(BaseLLMClient):
    """
    Async OpenAI chat completion client.

    Features:
    - Structured system / user message construction
    - Automatic retry with exponential back-off
    - Token usage tracking
    """

    def __init__(self) -> None:
        from openai import AsyncOpenAI

        settings = get_settings()
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")

        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_request_timeout,
        )
        self._model = settings.openai_chat_model
        self._max_tokens = settings.openai_max_tokens
        self._temperature = settings.openai_temperature

        logger.info("OpenAI LLM client initialised with model '%s'", self._model)

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(self, prompt: BuiltPrompt) -> LLMResponse:
        """
        Send the prompt to OpenAI and return a structured LLMResponse.
        Retries on rate-limit and transient API errors.
        """
        from openai import RateLimitError, APIStatusError, APIConnectionError

        messages = [
            {"role": "system", "content": prompt.system_message},
            {"role": "user", "content": prompt.user_message},
        ]

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(
                (RateLimitError, APIStatusError, APIConnectionError)
            ),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        ):
            with attempt:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "unknown",
        )


# ---------------------------------------------------------------------------
# Gemini client (future)
# ---------------------------------------------------------------------------

class GeminiLLMClient(BaseLLMClient):
    """Placeholder for Google Gemini LLM client."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "GeminiLLMClient is not yet implemented. "
            "Set LLM_PROVIDER=openai in your .env"
        )

    @property
    def model_name(self) -> str:
        return "gemini-1.5-pro"

    async def complete(self, prompt: BuiltPrompt) -> LLMResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_client() -> BaseLLMClient:
    """Instantiate the correct LLM client based on LLM_PROVIDER setting."""
    settings = get_settings()
    provider = settings.llm_provider

    if provider == LLMProvider.OPENAI:
        return OpenAILLMClient()
    elif provider == LLMProvider.GEMINI:
        return GeminiLLMClient()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_llm_client: BaseLLMClient | None = None


def get_llm_client() -> BaseLLMClient:
    """Return the module-level LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = create_llm_client()
    return _llm_client
