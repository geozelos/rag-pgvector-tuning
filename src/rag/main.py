"""
FastAPI application: chunk ingestion, vector similarity search, and physical-layer tuning.

**Flow**
    1. On startup, load YAML from ``config/`` (embedding dim, profiles, guardrails).
    2. Open an asyncpg pool to PostgreSQL (pgvector).
    3. HTTP handlers call :func:`rag.tuner.apply_search_session_params` so each retrieve
       uses the active profile's HNSW ``ef_search`` or IVFFlat ``probes`` (plus optional overrides).

**Endpoints** (see also ``/docs`` when the server runs)
    - ``GET /health`` — liveness.
    - ``GET /config/active-profile`` — active YAML profile and effective search params.
    - ``POST /ingest/chunks`` — upsert chunks with demo embeddings (deterministic, no external API).
    - ``POST /retrieve`` — k-NN order by cosine distance (pgvector ``<=>``).
    - ``GET /telemetry/summary`` — recent ingest/retrieve stats + table row count.
    - ``PATCH /config/runtime-search`` — set/clear in-memory overrides (whitelist-checked).
    - ``POST /tuner/recommend`` / ``POST /tuner/step`` — MVP tuner from in-process telemetry.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from rag.config_loader import AppYamlConfig, load_config_defaults
from rag.embeddings import demo_embedding
from rag.models_config import EmbeddingModelConfig
from rag.settings import Settings
from rag.telemetry import TelemetryCollector, assert_param_whitelisted
from rag.tuner import PhysicalTuner, apply_search_session_params

logger = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO, format="%(message)s")
_settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load config, create telemetry and tuner, connect DB pool; close pool on shutdown."""
    bundle = load_config_defaults()
    telemetry = TelemetryCollector()
    tuner = PhysicalTuner(profiles=bundle.profiles_doc, guardrails=bundle.guardrails)
    pool = await asyncpg.create_pool(_settings.database_url, min_size=1, max_size=10)
    app.state.bundle = bundle
    app.state.telemetry = telemetry
    app.state.tuner = tuner
    app.state.pool = pool
    yield
    await pool.close()


app = FastAPI(
    title="rag-pgvector-tuning",
    description="RAG retrieval demo: PostgreSQL pgvector + YAML-driven search parameter tuning.",
    lifespan=lifespan,
)


def bundle(req: Request) -> AppYamlConfig:
    return req.app.state.bundle


def embedding_cfg(req: Request) -> EmbeddingModelConfig:
    return bundle(req).embedding


def pool(req: Request) -> asyncpg.Pool:
    return req.app.state.pool


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a simple status object for load balancers or smoke tests."""
    return {"status": "ok"}


@app.get("/config/active-profile")
async def active_profile(req: Request) -> dict[str, Any]:
    """Expose ``active_profile`` from ``config/profiles.yaml`` and effective HNSW/IVF params."""
    tuner: PhysicalTuner = req.app.state.tuner
    name, profile = tuner.active()
    ef, probes = tuner.effective_search()
    return {
        "active_profile_name": name,
        "profile": profile.model_dump(),
        "effective_hnsw_ef_search": ef,
        "effective_ivfflat_probes": probes,
        "runtime_overrides": {
            "hnsw_ef_search": tuner.overrides.hnsw_ef_search,
            "ivfflat_probes": tuner.overrides.ivfflat_probes,
        },
    }


class ChunkIn(BaseModel):
    """One text chunk and metadata; ``doc_id`` + ``chunk_index`` form the upsert key."""

    tenant_id: str = Field(default="default")
    source_type: str = Field(default="doc")
    doc_id: str
    chunk_index: int = 0
    content: str


class IngestPayload(BaseModel):
    """Batch of chunks to embed and upsert into ``chunks``."""

    chunks: list[ChunkIn]


@app.post("/ingest/chunks")
async def ingest_chunks(req: Request, body: IngestPayload) -> dict[str, Any]:
    """Upsert chunks: compute demo vectors (see ``rag.embeddings``), INSERT ... ON CONFLICT."""
    cfg = embedding_cfg(req)
    pl = pool(req)
    telemetry: TelemetryCollector = req.app.state.telemetry
    if not body.chunks:
        raise HTTPException(status_code=400, detail="chunks must be non-empty")
    t0 = time.perf_counter()
    vals: list[tuple[Any, ...]] = []
    for row in body.chunks:
        emb = demo_embedding(row.content, cfg.embedding_dim)
        emb_lit = "[" + ",".join(f"{float(x):.8f}" for x in emb) + "]"
        vals.append((row.tenant_id, row.source_type, row.doc_id, row.chunk_index, row.content, emb_lit))
    elapsed_ms = 0.0
    try:
        async with pl.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO chunks (tenant_id, source_type, doc_id, chunk_index, content, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::vector)
                    ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
                      tenant_id = EXCLUDED.tenant_id,
                      source_type = EXCLUDED.source_type,
                      content = EXCLUDED.content,
                      embedding = EXCLUDED.embedding,
                      created_at = now();
                    """,
                    vals,
                )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        payload = telemetry.emit_ingest(
            batch_size=len(vals),
            duration_ms=elapsed_ms,
            failures=len(vals),
            tenant_id=body.chunks[0].tenant_id,
        )
        logger.warning("%s", TelemetryCollector.format_log_line(payload))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    payload = telemetry.emit_ingest(
        batch_size=len(vals),
        duration_ms=elapsed_ms,
        failures=0,
        tenant_id=body.chunks[0].tenant_id,
    )
    logger.info("%s", TelemetryCollector.format_log_line(payload))
    return {"upserted": len(vals), "duration_ms": elapsed_ms}


class RetrievePayload(BaseModel):
    """Search request: embed ``query`` with the same demo function used at ingest."""

    query: str = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=200)
    tenant_id: str | None = None
    source_type: str | None = None


@app.post("/retrieve")
async def retrieve(req: Request, body: RetrievePayload) -> dict[str, Any]:
    """Top-``k`` nearest neighbors by pgvector distance; applies session search params in a transaction."""
    cfg_e = embedding_cfg(req)
    pl = pool(req)
    tuner: PhysicalTuner = req.app.state.tuner
    telemetry: TelemetryCollector = req.app.state.telemetry
    name, profile = tuner.active()

    ef, probes = tuner.effective_search()
    t0 = time.perf_counter()
    query_vec = demo_embedding(body.query, cfg_e.embedding_dim)
    vec_lit = "[" + ",".join(f"{float(x):.8f}" for x in query_vec) + "]"

    conditions: list[str] = []
    params: list[Any] = [vec_lit, body.k]
    i = 3
    if body.tenant_id is not None:
        conditions.append(f"tenant_id = ${i}")
        params.append(body.tenant_id)
        i += 1
    if body.source_type is not None:
        conditions.append(f"source_type = ${i}")
        params.append(body.source_type)
        i += 1
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT id, doc_id, chunk_index, content,
               1 - (embedding <=> $1::vector) AS cosine_sim
        FROM chunks
        {where}
        ORDER BY embedding <=> $1::vector
        LIMIT $2
    """

    async with pl.acquire() as conn:
        async with conn.transaction():
            await apply_search_session_params(conn, profile, tuner.overrides)
            rows = await conn.fetch(sql, *params)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    payload = telemetry.emit_retrieve(
        duration_ms=elapsed_ms,
        k=body.k,
        index_family=profile.index_family,
        ef_search=ef if profile.index_family == "hnsw" else None,
        ivfflat_probes=probes if profile.index_family == "ivfflat" else None,
        filter_tenant_id=body.tenant_id,
        filter_source_type=body.source_type,
    )
    logger.info("%s", TelemetryCollector.format_log_line(payload))
    return {
        "profile": name,
        "index_family": profile.index_family,
        "hnsw_ef_search": ef if profile.index_family == "hnsw" else None,
        "ivfflat_probes": probes if profile.index_family == "ivfflat" else None,
        "duration_ms": elapsed_ms,
        "results": [dict(r) for r in rows],
    }


@app.get("/telemetry/summary")
async def telemetry_summary(req: Request) -> dict[str, Any]:
    """Rolling-window ingest/retrieve stats from memory plus ``SELECT count(*)`` on ``chunks``."""
    telemetry: TelemetryCollector = req.app.state.telemetry
    pl = pool(req)
    async with pl.acquire() as conn:
        n = await conn.fetchval("SELECT count(*)::bigint FROM chunks")
    summary = telemetry.summary()
    summary["corpus_chunks"] = int(n or 0)
    return summary


class RuntimeSearchPatch(BaseModel):
    """Patch in-memory overrides; values are clamped by guardrails when applied via tuner APIs."""

    hnsw_ef_search: int | None = None
    ivfflat_probes: int | None = None
    clear_overrides: bool = False


@app.patch("/config/runtime-search")
async def patch_runtime_search(req: Request, body: RuntimeSearchPatch) -> dict[str, Any]:
    """Set or clear ``hnsw_ef_search`` / ``ivfflat_probes`` overrides (must be whitelisted in YAML)."""
    tuner: PhysicalTuner = req.app.state.tuner
    g = bundle(req).guardrails
    if body.clear_overrides:
        tuner.clear_overrides()
        return {"overrides": tuner.overrides, "cleared": True}
    if body.hnsw_ef_search is not None:
        assert_param_whitelisted("hnsw.ef_search", g)
        tuner.set_override(hnsw_ef_search=body.hnsw_ef_search)
    if body.ivfflat_probes is not None:
        assert_param_whitelisted("ivfflat.probes", g)
        tuner.set_override(ivfflat_probes=body.ivfflat_probes)
    return {"overrides": tuner.overrides}


@app.post("/tuner/recommend")
async def tuner_recommend(req: Request) -> dict[str, Any]:
    """Suggest increase/decrease/hold for search params from ``TelemetryCollector.summary()``."""
    tuner: PhysicalTuner = req.app.state.tuner
    telemetry: TelemetryCollector = req.app.state.telemetry
    return tuner.recommend(telemetry)


@app.post("/tuner/step")
async def tuner_step(
    req: Request,
    auto_apply: bool = Query(default=False),
) -> dict[str, Any]:
    """Run :meth:`rag.tuner.PhysicalTuner.maybe_apply_from_recommendation` (optional apply)."""
    tuner: PhysicalTuner = req.app.state.tuner
    telemetry: TelemetryCollector = req.app.state.telemetry
    return tuner.maybe_apply_from_recommendation(telemetry, auto_apply=auto_apply)


@app.post("/telemetry/ingest-backlog/clear")
async def clear_ingest_backlog(req: Request) -> dict[str, str]:
    """Reset the heuristic ingest-backlog counter used by the tuner (dev/demo helper)."""
    telemetry: TelemetryCollector = req.app.state.telemetry
    telemetry.ingest_batches_pending_estimate = 0.0
    return {"status": "cleared"}
