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
