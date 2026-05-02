"""HTTP API tests with a mocked PostgreSQL pool (no database required)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient


from rag.config_loader import AppYamlConfig


def test_health_ok(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_ok(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db
    conn.fetchval = AsyncMock(side_effect=[1, True, True])
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ready", "checks": {"database": True, "pgvector": True, "chunks_table": True}}


def test_ready_503_when_pgvector_missing(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db
    conn.fetchval = AsyncMock(side_effect=[1, False, True])
    r = client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["pgvector"] is False
    assert "failed" in body["detail"]


def test_ready_503_when_chunks_table_missing(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db
    conn.fetchval = AsyncMock(side_effect=[1, True, False])
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["chunks_table"] is False


def test_ready_503_on_db_error(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db
    conn.fetchval = AsyncMock(side_effect=RuntimeError("connection refused"))
    r = client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert "connection refused" in body["detail"]


def test_ingest_chunks_exceeds_limit_returns_413(
    http_client_mock_db: tuple[TestClient, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    import rag.main as main_mod

    monkeypatch.setattr(main_mod._settings, "max_ingest_chunks_per_request", 2)
    client, _conn = http_client_mock_db
    r = client.post(
        "/ingest/chunks",
        json={
            "chunks": [
                {"doc_id": "a", "chunk_index": 0, "content": "x"},
                {"doc_id": "b", "chunk_index": 0, "content": "y"},
                {"doc_id": "c", "chunk_index": 0, "content": "z"},
            ],
        },
    )
    assert r.status_code == 413
    assert "exceeds limit 2" in r.json()["detail"]


def test_ingest_requires_api_key_when_configured(http_client_mock_db_with_api_key: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db_with_api_key
    r = client.post(
        "/ingest/chunks",
        json={
            "chunks": [{"doc_id": "d1", "chunk_index": 0, "content": "hello"}],
        },
    )
    assert r.status_code == 401
    assert r.json()["detail"] == "invalid or missing API key"


def test_ingest_with_bearer_api_key(http_client_mock_db_with_api_key: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db_with_api_key
    r = client.post(
        "/ingest/chunks",
        headers={"Authorization": "Bearer test-secret-key"},
        json={
            "chunks": [{"doc_id": "d1", "chunk_index": 0, "content": "hello"}],
        },
    )
    assert r.status_code == 200
    assert conn.executemany.await_count >= 1


def test_ingest_with_x_api_key_header(http_client_mock_db_with_api_key: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db_with_api_key
    r = client.post(
        "/ingest/chunks",
        headers={"X-API-Key": "test-secret-key"},
        json={
            "chunks": [{"doc_id": "d1", "chunk_index": 0, "content": "hello"}],
        },
    )
    assert r.status_code == 200
    assert conn.executemany.await_count >= 1


def test_health_and_ready_skip_api_key(http_client_mock_db_with_api_key: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db_with_api_key
    assert client.get("/health").status_code == 200
    conn.fetchval = AsyncMock(side_effect=[1, True, True])
    assert client.get("/ready").status_code == 200


def test_active_profile(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    r = client.get("/config/active-profile")
    assert r.status_code == 200
    body = r.json()
    assert "active_profile_name" in body
    assert body["effective_hnsw_ef_search"] is not None


def test_ingest_empty_chunks_400(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    r = client.post("/ingest/chunks", json={"chunks": []})
    assert r.status_code == 400


def test_ingest_success(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db
    r = client.post(
        "/ingest/chunks",
        json={
            "chunks": [
                {
                    "tenant_id": "t1",
                    "source_type": "doc",
                    "doc_id": "d1",
                    "chunk_index": 0,
                    "content": "hello world",
                },
            ],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["upserted"] == 1
    assert "duration_ms" in data
    assert conn.executemany.await_count >= 1


def test_ingest_db_error_400(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db
    conn.executemany = AsyncMock(side_effect=RuntimeError("forced db error"))
    r = client.post(
        "/ingest/chunks",
        json={
            "chunks": [
                {
                    "doc_id": "d2",
                    "chunk_index": 0,
                    "content": "x",
                },
            ],
        },
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "Ingest failed (see server logs)."


def test_retrieve_returns_results(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db
    row = {
        "id": 1,
        "doc_id": "d1",
        "chunk_index": 0,
        "content": "c1",
        "cosine_sim": 0.9,
    }
    conn.fetch = AsyncMock(return_value=[row])
    r = client.post("/retrieve", json={"query": "q", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["profile"]
    assert body["index_family"] == "hnsw"
    assert len(body["results"]) == 1
    assert body["results"][0]["doc_id"] == "d1"
    assert conn.fetch.await_count >= 1


def test_retrieve_with_filters(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db
    conn.fetch = AsyncMock(return_value=[])
    r = client.post(
        "/retrieve",
        json={"query": "q", "k": 5, "tenant_id": "t", "source_type": "doc"},
    )
    assert r.status_code == 200
    call_args = conn.fetch.call_args
    sql = call_args[0][0]
    assert "tenant_id" in sql
    assert "source_type" in sql


def test_retrieve_tenant_id_only(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db
    conn.fetch = AsyncMock(return_value=[])
    r = client.post("/retrieve", json={"query": "q", "k": 5, "tenant_id": "t"})
    assert r.status_code == 200
    sql = conn.fetch.call_args[0][0]
    assert "WHERE tenant_id = $3" in sql.replace("\n", " ")
    assert "source_type = $" not in sql


def test_retrieve_source_type_only(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db
    conn.fetch = AsyncMock(return_value=[])
    r = client.post("/retrieve", json={"query": "q", "k": 5, "source_type": "doc"})
    assert r.status_code == 200
    sql = conn.fetch.call_args[0][0]
    assert "WHERE source_type = $3" in sql.replace("\n", " ")
    assert "tenant_id = $" not in sql


def test_telemetry_summary(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, conn = http_client_mock_db
    conn.fetchval = AsyncMock(return_value=42)
    r = client.get("/telemetry/summary")
    assert r.status_code == 200
    assert r.json()["corpus_chunks"] == 42


def test_patch_runtime_clear(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    client.patch("/config/runtime-search", json={"hnsw_ef_search": 48})
    r = client.patch("/config/runtime-search", json={"clear_overrides": True})
    assert r.status_code == 200
    assert r.json().get("cleared") is True


def test_patch_runtime_hnsw_override(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    r = client.patch("/config/runtime-search", json={"hnsw_ef_search": 44})
    assert r.status_code == 200
    assert r.json()["overrides"]["hnsw_ef_search"] == 44


def test_tuner_recommend_and_step(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    for _ in range(80):
        client.post("/retrieve", json={"query": "warmup", "k": 10})
    rec = client.post("/tuner/recommend")
    assert rec.status_code == 200
    assert "action" in rec.json()
    step = client.post("/tuner/step", params={"auto_apply": False})
    assert step.status_code == 200


def test_tuner_step_valueerror_returns_400(
    http_client_mock_db: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _conn = http_client_mock_db
    tuner = client.app.state.tuner

    def _boom(*_a: object, **_k: object) -> None:
        raise ValueError("simulated")

    monkeypatch.setattr(tuner, "maybe_apply_from_recommendation", _boom)
    r = client.post("/tuner/step", params={"auto_apply": False})
    assert r.status_code == 400
    assert r.json()["detail"] == "Invalid tuner operation."


def test_patch_runtime_ivfflat_probes(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    r = client.patch("/config/runtime-search", json={"ivfflat_probes": 12})
    assert r.status_code == 200
    assert r.json()["overrides"]["ivfflat_probes"] == 12


def test_patch_runtime_hnsw_out_of_bounds(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    r = client.patch("/config/runtime-search", json={"hnsw_ef_search": 8})
    assert r.status_code == 400
    assert r.json()["detail"] == "Invalid runtime search override."


def test_patch_runtime_ivfflat_probes_out_of_bounds(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    r = client.patch("/config/runtime-search", json={"ivfflat_probes": 99})
    assert r.status_code == 400
    assert r.json()["detail"] == "Invalid runtime search override."


def test_patch_runtime_whitelist_denied(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    b = client.app.state.bundle
    client.app.state.bundle = AppYamlConfig(
        embedding=b.embedding,
        profiles_doc=b.profiles_doc,
        guardrails=b.guardrails.model_copy(update={"whitelist_runtime_params": []}),
    )
    r = client.patch("/config/runtime-search", json={"hnsw_ef_search": 44})
    assert r.status_code == 400
    assert r.json()["detail"] == "Invalid runtime search override."


def test_ingest_chunk_content_too_long(
    http_client_mock_db: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rag.main as main_mod

    monkeypatch.setattr(main_mod._settings, "max_chunk_content_chars", 5)
    client, _conn = http_client_mock_db
    r = client.post(
        "/ingest/chunks",
        json={"chunks": [{"doc_id": "d", "chunk_index": 0, "content": "123456"}]},
    )
    assert r.status_code == 422


def test_retrieve_query_too_long(
    http_client_mock_db: tuple[TestClient, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rag.main as main_mod

    monkeypatch.setattr(main_mod._settings, "max_retrieve_query_chars", 5)
    client, conn = http_client_mock_db
    conn.fetch = AsyncMock(return_value=[])
    r = client.post("/retrieve", json={"query": "123456", "k": 3})
    assert r.status_code == 422


def test_clear_ingest_backlog(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    r = client.post("/telemetry/ingest-backlog/clear")
    assert r.status_code == 200
    assert r.json()["status"] == "cleared"
