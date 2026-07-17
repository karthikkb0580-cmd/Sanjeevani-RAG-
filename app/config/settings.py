"""
Module 2: Configuration
app/config/settings.py

Centralised settings loaded from environment variables via Pydantic-Settings.
Supports multiple embedding providers and LLM backends.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Provider Enums
# ---------------------------------------------------------------------------

class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"       # future


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    NEMOTRON = "nemotron"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = Field(default="Sanjeevani RAG Service")
    app_version: str = Field(default="1.0.0")
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")

    # ── API Server ────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # ── Active Providers ─────────────────────────────────────────────────────
    embedding_provider: EmbeddingProvider = Field(default=EmbeddingProvider.OPENAI)
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI)

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="")
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    openai_embedding_dimensions: int = Field(default=1536)
    openai_chat_model: str = Field(default="gpt-4o")
    openai_max_tokens: int = Field(default=4096)
    openai_temperature: float = Field(default=0.2)
    openai_request_timeout: int = Field(default=60)

    # ── Gemini (Embeddings only) ───────────────────────────────────────────────
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-3.5-flash")
    gemini_embedding_model: str = Field(default="models/text-embedding-004")
    gemini_embedding_dimensions: int = Field(default=768)

    # ── NVIDIA NIM API (Text LLM + Vision) ────────────────────────────────────
    nvidia_api_key: str = Field(default="")
    # Text LLM (Nemotron)
    nemotron_model: str = Field(default="nvidia/nemotron-ultra-253b-v1")
    nemotron_max_tokens: int = Field(default=4096)
    nemotron_temperature: float = Field(default=0.2)
    nemotron_request_timeout: int = Field(default=120)
    # Vision model — used by the multipart endpoint for image→molecule extraction
    # Recommended: nvidia/llama-3.2-90b-vision-instruct (highest accuracy)
    #              nvidia/llama-3.2-11b-vision-instruct  (faster / lower cost)
    nvidia_vision_model: str = Field(default="nvidia/llama-3.2-90b-vision-instruct")
    nvidia_vision_max_tokens: int = Field(default=1024)
    nvidia_vision_temperature: float = Field(default=0.1)

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_grpc_port: int = Field(default=6334)
    qdrant_api_key: str = Field(default="")
    qdrant_collection_name: str = Field(default="research_documents")
    qdrant_use_https: bool = Field(default=False)

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=600)
    chunk_overlap: int = Field(default=100)
    chunk_min_size: int = Field(default=50)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=10)
    retrieval_similarity_threshold: float = Field(default=0.65)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    reranker_top_n: int = Field(default=5)

    # ── File Upload ───────────────────────────────────────────────────────────
    max_file_size_mb: int = Field(default=50)
    allowed_extensions: str = Field(default="pdf,txt,docx")
    upload_dir: str = Field(default="/tmp/rag_uploads")

    # ── Vision / Image Upload ─────────────────────────────────────────────────
    max_image_size_mb: int = Field(default=20)
    allowed_image_mimes: str = Field(
        default="image/jpeg,image/png,image/webp,image/gif,image/bmp,image/tiff"
    )

    @property
    def allowed_image_mime_set(self) -> set[str]:
        return {m.strip().lower() for m in self.allowed_image_mimes.split(",")}

    # ── Derived Properties ────────────────────────────────────────────────────

    @property
    def allowed_extension_set(self) -> set[str]:
        return {ext.strip().lower() for ext in self.allowed_extensions.split(",")}

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def qdrant_url(self) -> str:
        scheme = "https" if self.qdrant_use_https else "http"
        return f"{scheme}://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def embedding_dimensions(self) -> int:
        """Return the vector dimension for the active embedding provider."""
        if self.embedding_provider == EmbeddingProvider.OPENAI:
            return self.openai_embedding_dimensions
        elif self.embedding_provider == EmbeddingProvider.GEMINI:
            return self.gemini_embedding_dimensions
        return self.openai_embedding_dimensions

    @field_validator("openai_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError("openai_temperature must be between 0.0 and 2.0")
        return v

    @field_validator("retrieval_similarity_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("retrieval_similarity_threshold must be between 0.0 and 1.0")
        return v

    def ensure_upload_dir(self) -> Path:
        """Create upload directory if it does not exist."""
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()
