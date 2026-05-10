"""Tests for Open Core features: tenant enforcement, metadata filters, CORS, OpenAPI UI toggle, rate limits."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from rag.main import create_app
from rag.settings import Settings

from tests.conftest import FakeConn, FakePool


def test_retrieve_requires_tenant_when_configured(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    client.app.state.settings.require_tenant_id = True
    try:
        r = client.post("/retrieve", json={"query": "hello", "k": 3})
        assert r.status_code == 400
        assert "tenant_id" in r.json()["detail"]
    finally:
        client.app.state.settings.require_tenant_id = False


def test_retrieve_metadata_filter_sql(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db
    conn.fetch = AsyncMock(return_value=[])
    r = client.post(
        "/retrieve",
        json={"query": "q", "k": 5, "metadata_filter": {"topic": "faq"}},
    )
    assert r.status_code == 200
    sql = conn.fetch.call_args[0][0]
    assert "metadata @>" in sql


def test_openapi_swagger_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag.main as main_mod

    monkeypatch.delenv("DISABLE_OPENAPI_UI", raising=False)

    conn = FakeConn()

    async def _create_pool(*_a: object, **_k: object) -> FakePool:
        return FakePool(conn)

    monkeypatch.setattr(main_mod, "asyncpg", SimpleNamespace(create_pool=_create_pool))

    app = create_app(Settings(disable_openapi_ui=True))
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404


def test_default_health_without_cors_allow_origin(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _ = http_client_mock_db
    r = client.get("/health", headers={"Origin": "https://frontend.example"})
    assert r.headers.get("access-control-allow-origin") is None


def test_cors_allowlist_sets_allow_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag.main as main_mod

    conn = FakeConn()

    async def _create_pool(*_a: object, **_k: object) -> FakePool:
        return FakePool(conn)

    monkeypatch.setattr(main_mod, "asyncpg", SimpleNamespace(create_pool=_create_pool))

    settings = Settings(cors_origins="https://frontend.example")
    app = create_app(settings)
    with TestClient(app) as client:
        r = client.get("/health", headers={"Origin": "https://frontend.example"})
        assert r.headers.get("access-control-allow-origin") == "https://frontend.example"


@pytest.fixture
def client_rate_limit_one(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, FakeConn]:
    import rag.main as main_mod

    conn = FakeConn()

    async def _create_pool(*_a: object, **_k: object) -> FakePool:
        return FakePool(conn)

    monkeypatch.setattr(main_mod, "asyncpg", SimpleNamespace(create_pool=_create_pool))

    settings = Settings(rate_limit_per_minute=1, database_url="postgresql://mock/mock")
    app = main_mod.create_app(settings)
    with TestClient(app) as client:
        yield client, conn


def test_rate_limit_applies_to_api_routes(client_rate_limit_one: tuple[TestClient, FakeConn]) -> None:
    client, conn = client_rate_limit_one
    conn.fetchval = AsyncMock(return_value=0)
    assert client.get("/health").status_code == 200
    assert client.get("/telemetry/summary").status_code == 200
    assert client.get("/telemetry/summary").status_code == 429
