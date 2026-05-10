"""
FastAPI application: chunk ingestion, vector similarity search, and physical-layer tuning.

**Flow**
    1. On startup, load YAML from ``config/`` (embedding dim, profiles, guardrails).
    2. Open an asyncpg pool to PostgreSQL (pgvector).
    3. HTTP handlers call :func:`rag.tuner.apply_search_session_params` so each retrieve
       uses the active profile's HNSW ``ef_search`` or IVFFlat ``probes`` (plus optional overrides).

**Endpoints** (see also ``/docs`` when the server runs)
    - ``GET /health`` — liveness (process only; does not query the database).
    - ``GET /ready`` — readiness: database connectivity, pgvector extension, ``chunks`` table.
    - ``GET /config/active-profile`` — active YAML profile and effective search params.
    - ``POST /ingest/chunks`` — upsert chunks; vectors from configured :mod:`rag.embedding_backend` (demo hash, OpenAI, or local HTTP).
    - ``POST /retrieve`` — k-NN order by cosine distance (pgvector ``<=>``).
    - ``GET /telemetry/summary`` — recent ingest/retrieve stats + table row count.
    - ``PATCH /config/runtime-search`` — set/clear in-memory overrides (whitelist-checked).
    - ``POST /tuner/recommend`` / ``POST /tuner/step`` — MVP tuner from in-process telemetry.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from typing import Any, AsyncIterator

import asyncpg
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from rag import __version__ as _package_version
from rag.config_loader import AppYamlConfig, load_config_defaults
from rag.embedding_backend import build_embedding_backend
from rag.models_config import EmbeddingModelConfig
from rag.settings import Settings
from rag.telemetry import TelemetryCollector, assert_param_whitelisted
from rag.tuner import PhysicalTuner, apply_search_session_params

logger = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO, format="%(message)s")
_settings = Settings()

router = APIRouter()

_rate_limit_window_s = 60.0


def create_app(settings: Settings) -> FastAPI:
    """Construct the ASGI app with middleware and routes bound to ``settings``."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load config, create telemetry and tuner, connect DB pool; close pool on shutdown."""
        bundle = load_config_defaults()
        telemetry = TelemetryCollector()
        tuner = PhysicalTuner(profiles=bundle.profiles_doc, guardrails=bundle.guardrails)
        emb_backend = build_embedding_backend(settings)
        if settings.embedding_backend.strip().lower() == "openai" and settings.openai_api_key is None:
            await emb_backend.aclose()
            raise RuntimeError("OPENAI_API_KEY is required when EMBEDDING_BACKEND=openai")
        try:
            pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=10)
        except Exception:
            await emb_backend.aclose()
            raise
        app.state.bundle = bundle
        app.state.telemetry = telemetry
        app.state.tuner = tuner
        app.state.embedding_backend = emb_backend
        app.state.pool = pool
        yield
        await emb_backend.aclose()
        await pool.close()

    docs_url = None if settings.disable_openapi_ui else "/docs"
    redoc_url = None if settings.disable_openapi_ui else "/redoc"

    app = FastAPI(
        title="rag-pgvector-tuning",
        description="RAG retrieval demo: PostgreSQL pgvector + YAML-driven search parameter tuning.",
        lifespan=lifespan,
        version=_package_version,
        docs_url=docs_url,
        redoc_url=redoc_url,
    )
    app.state.settings = settings
    app.state.rate_limit_buckets = defaultdict(deque)

    if (settings.cors_origins or "").strip():
        cors_list = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
        if cors_list:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_list,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        """Best-effort per-IP limit when ``RATE_LIMIT_PER_MINUTE`` is set (single-process only)."""
        lim = settings.rate_limit_per_minute
        if lim is None:
            return await call_next(request)
        path = request.url.path
        if path in ("/health", "/ready"):
            return await call_next(request)
        peer = request.client.host if request.client else "unknown"
        now = time.monotonic()
        buckets: dict[str, deque[float]] = request.app.state.rate_limit_buckets
        dq = buckets[peer]
        while dq and now - dq[0] > _rate_limit_window_s:
            dq.popleft()
        if len(dq) >= lim:
            return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
        dq.append(now)
        return await call_next(request)

    @app.middleware("http")
    async def optional_api_key_middleware(request: Request, call_next):
        """When ``RAG_API_KEY`` is set, require credentials on all routes except probes.

        If ``Authorization`` starts with ``Bearer `` (case-insensitive), only the Bearer
        token is accepted (empty token after the prefix is rejected; ``X-API-Key`` is
        not consulted). Otherwise ``X-API-Key`` alone may authenticate.
        """
        path = request.url.path
        if path in ("/health", "/ready"):
            return await call_next(request)
        expected = settings.api_key
        if expected is None:
            return await call_next(request)
        auth_header = request.headers.get("Authorization") or ""
        lower = auth_header.lower()
        if lower.startswith("bearer "):
            bearer_token = auth_header[7:].strip()
            if not bearer_token:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "invalid or missing API key"},
                )
            candidate = bearer_token
        else:
            candidate = request.headers.get("X-API-Key")
        if candidate != expected:
            return JSONResponse(
                status_code=401,
                content={"detail": "invalid or missing API key"},
            )
        return await call_next(request)

    app.include_router(router)
    return app


def bundle(req: Request) -> AppYamlConfig:
    return req.app.state.bundle


def embedding_cfg(req: Request) -> EmbeddingModelConfig:
    return bundle(req).embedding


def pool(req: Request) -> asyncpg.Pool:
    return req.app.state.pool


def embedding_backend(req: Request):
    return req.app.state.embedding_backend


def _build_retrieve_sql_and_params(
    vec_lit: str,
    k: int,
    tenant_id: str | None,
    source_type: str | None,
    metadata_filter: dict[str, Any] | None,
) -> tuple[str, list[Any]]:
    """Compose WHERE clauses for tenant/source/metadata containment; parameters start at ``$3``."""
    where_parts: list[str] = []
    params: list[Any] = [vec_lit, k]
    idx = 3
    if tenant_id is not None:
        where_parts.append(f"tenant_id = ${idx}")
        params.append(tenant_id)
        idx += 1
    if source_type is not None:
        where_parts.append(f"source_type = ${idx}")
        params.append(source_type)
        idx += 1
    if metadata_filter:
        where_parts.append(f"metadata @> ${idx}::jsonb")
        params.append(json.dumps(metadata_filter))
        idx += 1
    where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    # WHERE fragments are fixed templates; values are passed only via asyncpg parameters.
    sql = (
        "SELECT id, doc_id, chunk_index, content, metadata, "  # nosec B608
        "1 - (embedding <=> $1::vector) AS cosine_sim "
        "FROM chunks"
        + where_sql
        + " ORDER BY embedding <=> $1::vector LIMIT $2"
    )
    return sql, params


@router.get("/health")
async def health() -> dict[str, str]:
    """Return a simple status object for load balancers or smoke tests."""
    return {"status": "ok"}


@router.get("/ready", response_model=None)
async def ready(req: Request) -> JSONResponse | dict[str, Any]:
    """Verify PostgreSQL, pgvector extension, and migrated ``chunks`` table (use for readiness probes)."""
    checks: dict[str, bool] = {"database": False, "pgvector": False, "chunks_table": False}
    pl = pool(req)
    try:
        async with pl.acquire() as conn:
            await conn.fetchval("SELECT 1")
            checks["database"] = True
            ext_ok = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
            checks["pgvector"] = bool(ext_ok)
            tbl_ok = await conn.fetchval("SELECT to_regclass('public.chunks') IS NOT NULL")
            checks["chunks_table"] = bool(tbl_ok)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ready check failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "checks": checks,
                "detail": str(exc),
            },
        )

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "checks": checks,
                "detail": f"checks failed: {', '.join(failed)}",
            },
        )
    return {"status": "ready", "checks": checks}


@router.get("/config/active-profile")
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
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestPayload(BaseModel):
    """Batch of chunks to embed and upsert into ``chunks``."""

    chunks: list[ChunkIn]


@router.post("/ingest/chunks")
async def ingest_chunks(req: Request, body: IngestPayload) -> dict[str, Any]:
    """Upsert chunks: compute vectors via the configured embedding backend, INSERT ... ON CONFLICT."""
    cfg = embedding_cfg(req)
    pl = pool(req)
    telemetry: TelemetryCollector = req.app.state.telemetry
    backend = embedding_backend(req)
    if not body.chunks:
        raise HTTPException(status_code=400, detail="chunks must be non-empty")
    max_chunks = req.app.state.settings.max_ingest_chunks_per_request
    if len(body.chunks) > max_chunks:
        raise HTTPException(
            status_code=413,
            detail=f"chunks length {len(body.chunks)} exceeds limit {max_chunks} (configure MAX_INGEST_CHUNKS_PER_REQUEST)",
        )
    lim_content = req.app.state.settings.max_chunk_content_chars
    for row in body.chunks:
        if len(row.content) > lim_content:
            raise HTTPException(
                status_code=422,
                detail=f"content exceeds maximum length ({lim_content} characters)",
            )
    t0 = time.perf_counter()
    vals: list[tuple[Any, ...]] = []
    try:
        for row in body.chunks:
            emb = await backend.embed(row.content, cfg.embedding_dim)
            emb_lit = "[" + ",".join(f"{float(x):.8f}" for x in emb) + "]"
            meta_json = json.dumps(row.metadata) if row.metadata else "{}"
            vals.append(
                (
                    row.tenant_id,
                    row.source_type,
                    row.doc_id,
                    row.chunk_index,
                    row.content,
                    meta_json,
                    emb_lit,
                ),
            )
    except (RuntimeError, ValueError) as exc:
        logger.warning("embedding failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Embedding provider failed (see server logs).",
        ) from None
    elapsed_ms = 0.0
    try:
        async with pl.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO chunks (tenant_id, source_type, doc_id, chunk_index, content, metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::vector)
                    ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
                      tenant_id = EXCLUDED.tenant_id,
                      source_type = EXCLUDED.source_type,
                      content = EXCLUDED.content,
                      metadata = EXCLUDED.metadata,
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
        logger.warning("ingest failed: %s", exc, exc_info=True)
        logger.warning("%s", TelemetryCollector.format_log_line(payload))
        raise HTTPException(
            status_code=400,
            detail="Ingest failed (see server logs).",
        ) from None
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
    """Search request: embed ``query`` with the same backend used at ingest."""

    query: str = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=200)
    tenant_id: str | None = None
    source_type: str | None = None
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="JSON object matched with ``metadata @> filter`` (AND semantics on keys present).",
    )


@router.post("/retrieve")
async def retrieve(req: Request, body: RetrievePayload) -> dict[str, Any]:
    """Top-``k`` nearest neighbors by pgvector distance; applies session search params in a transaction."""
    lim_q = req.app.state.settings.max_retrieve_query_chars
    if len(body.query) > lim_q:
        raise HTTPException(
            status_code=422,
            detail=f"query exceeds maximum length ({lim_q} characters)",
        )
    if req.app.state.settings.require_tenant_id:
        tid = body.tenant_id
        if tid is None or (isinstance(tid, str) and tid.strip() == ""):
            raise HTTPException(
                status_code=400,
                detail="tenant_id is required when REQUIRE_TENANT_ID=true",
            )

    cfg_e = embedding_cfg(req)
    pl = pool(req)
    tuner: PhysicalTuner = req.app.state.tuner
    telemetry: TelemetryCollector = req.app.state.telemetry
    backend = embedding_backend(req)
    name, profile = tuner.active()

    ef, probes = tuner.effective_search()
    t0 = time.perf_counter()
    try:
        query_vec = await backend.embed(body.query, cfg_e.embedding_dim)
    except (RuntimeError, ValueError) as exc:
        logger.warning("query embedding failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Embedding provider failed (see server logs).",
        ) from None
    vec_lit = "[" + ",".join(f"{float(x):.8f}" for x in query_vec) + "]"

    sql, sql_params = _build_retrieve_sql_and_params(
        vec_lit,
        body.k,
        body.tenant_id,
        body.source_type,
        body.metadata_filter,
    )

    async with pl.acquire() as conn:
        async with conn.transaction():
            await apply_search_session_params(conn, profile, tuner.overrides)
            rows = await conn.fetch(sql, *sql_params)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    payload = telemetry.emit_retrieve(
        duration_ms=elapsed_ms,
        k=body.k,
        index_family=profile.index_family,
        ef_search=ef if profile.index_family == "hnsw" else None,
        ivfflat_probes=probes if profile.index_family == "ivfflat" else None,
        filter_tenant_id=body.tenant_id,
        filter_source_type=body.source_type,
        filter_metadata=body.metadata_filter,
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


@router.get("/telemetry/summary")
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


@router.patch("/config/runtime-search")
async def patch_runtime_search(req: Request, body: RuntimeSearchPatch) -> dict[str, Any]:
    """Set or clear ``hnsw_ef_search`` / ``ivfflat_probes`` overrides (must be whitelisted in YAML)."""
    tuner: PhysicalTuner = req.app.state.tuner
    g = bundle(req).guardrails
    if body.clear_overrides:
        tuner.clear_overrides()
        return {"overrides": tuner.overrides, "cleared": True}
    try:
        if body.hnsw_ef_search is not None:
            assert_param_whitelisted("hnsw.ef_search", g)
            tuner.set_override(hnsw_ef_search=body.hnsw_ef_search)
        if body.ivfflat_probes is not None:
            assert_param_whitelisted("ivfflat.probes", g)
            tuner.set_override(ivfflat_probes=body.ivfflat_probes)
    except ValueError as exc:
        logger.warning("runtime-search patch rejected: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Invalid runtime search override.",
        ) from None
    return {"overrides": tuner.overrides}


@router.post("/tuner/recommend")
async def tuner_recommend(req: Request) -> dict[str, Any]:
    """Suggest increase/decrease/hold for search params from ``TelemetryCollector.summary()``."""
    tuner: PhysicalTuner = req.app.state.tuner
    telemetry: TelemetryCollector = req.app.state.telemetry
    return tuner.recommend(telemetry)


@router.post("/tuner/step")
async def tuner_step(
    req: Request,
    auto_apply: bool = Query(default=False),
) -> dict[str, Any]:
    """Run :meth:`rag.tuner.PhysicalTuner.maybe_apply_from_recommendation` (optional apply)."""
    tuner: PhysicalTuner = req.app.state.tuner
    telemetry: TelemetryCollector = req.app.state.telemetry
    try:
        return tuner.maybe_apply_from_recommendation(telemetry, auto_apply=auto_apply)
    except ValueError as exc:
        logger.warning("tuner step rejected: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Invalid tuner operation.",
        ) from None


@router.post("/telemetry/ingest-backlog/clear")
async def clear_ingest_backlog(req: Request) -> dict[str, str]:
    """Reset the heuristic ingest-backlog counter used by the tuner (dev/demo helper)."""
    telemetry: TelemetryCollector = req.app.state.telemetry
    telemetry.ingest_batches_pending_estimate = 0.0
    return {"status": "cleared"}


app = create_app(_settings)
