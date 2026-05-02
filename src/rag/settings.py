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

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_api_key_as_none(cls, value: object) -> object:
        if value == "":
            return None
        return value
