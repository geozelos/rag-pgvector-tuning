"""Settings / environment loading."""

from __future__ import annotations

import pytest

from rag.settings import Settings


def test_settings_database_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/dbname")
    s = Settings()
    assert s.database_url == "postgresql://u:p@db:5432/dbname"


def test_settings_default_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    s = Settings()
    assert "postgresql" in s.database_url
    assert "5433" in s.database_url or "5432" in s.database_url


def test_settings_api_key_empty_string_becomes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", "")
    s = Settings()
    assert s.api_key is None


def test_settings_api_key_non_empty_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", "my-key")
    s = Settings()
    assert s.api_key == "my-key"


def test_settings_max_ingest_chunks_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAX_INGEST_CHUNKS_PER_REQUEST", raising=False)
    s = Settings()
    assert s.max_ingest_chunks_per_request == 500
