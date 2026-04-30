"""HTTP API tests with a mocked PostgreSQL pool (no database required)."""

from __future__ import annotations

from unittest.mock import AsyncMock

from starlette.testclient import TestClient


def test_health_ok(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


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
    assert "forced db error" in r.json()["detail"]


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


def test_patch_runtime_ivfflat_probes(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    r = client.patch("/config/runtime-search", json={"ivfflat_probes": 12})
    assert r.status_code == 200
    assert r.json()["overrides"]["ivfflat_probes"] == 12


def test_clear_ingest_backlog(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _conn = http_client_mock_db
    r = client.post("/telemetry/ingest-backlog/clear")
    assert r.status_code == 200
    assert r.json()["status"] == "cleared"
