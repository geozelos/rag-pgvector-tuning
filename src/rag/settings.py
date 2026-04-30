"""Application settings from environment (see ``DATABASE_URL``) and optional ``.env`` file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strip of settings loaded at import time in :mod:`rag.main` (Postgres DSN)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://rag:rag@localhost:5433/rag"
