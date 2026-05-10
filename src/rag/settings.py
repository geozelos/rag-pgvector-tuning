"""Application settings from environment (see ``DATABASE_URL``) and optional ``.env`` file."""

from __future__ import annotations

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strip of settings loaded at import time in :mod:`rag.main` (Postgres DSN)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str = "postgresql://rag:rag@localhost:5433/rag"
    max_ingest_chunks_per_request: int = Field(default=500, ge=1)
    max_chunk_content_chars: int = Field(default=500_000, ge=1)
    max_retrieve_query_chars: int = Field(default=32_768, ge=1)
    api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RAG_API_KEY", "API_KEY"),
    )

    embedding_backend: str = Field(
        default="demo",
        validation_alias=AliasChoices("EMBEDDING_BACKEND"),
        description="demo (hash), openai (HTTP API), or local (OpenAI-compatible HTTP, e.g. Ollama/TEI).",
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY"),
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("OPENAI_BASE_URL"),
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("OPENAI_EMBEDDING_MODEL"),
    )
    local_embeddings_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("LOCAL_EMBEDDINGS_BASE_URL"),
    )
    local_embeddings_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LOCAL_EMBEDDINGS_API_KEY"),
    )
    local_embedding_model: str = Field(
        default="nomic-embed-text",
        validation_alias=AliasChoices("LOCAL_EMBEDDING_MODEL"),
    )
    embedding_http_timeout_s: float = Field(
        default=60.0,
        ge=1.0,
        validation_alias=AliasChoices("EMBEDDING_HTTP_TIMEOUT_S"),
    )

    require_tenant_id: bool = Field(
        default=False,
        validation_alias=AliasChoices("REQUIRE_TENANT_ID"),
    )
    cors_origins: str | None = Field(
        default=None,
        validation_alias=AliasChoices("CORS_ORIGINS"),
        description="Comma-separated origins for browser clients; unset keeps FastAPI defaults.",
    )
    disable_openapi_ui: bool = Field(
        default=False,
        validation_alias=AliasChoices("DISABLE_OPENAPI_UI"),
    )
    rate_limit_per_minute: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("RATE_LIMIT_PER_MINUTE"),
        description="Best-effort in-process limit per client IP; unset disables (use a proxy in production).",
    )

    @field_validator("api_key", "openai_api_key", "local_embeddings_api_key", mode="before")
    @classmethod
    def _empty_secret_as_none(cls, value: object) -> object:
        if value == "":
            return None
        return value
