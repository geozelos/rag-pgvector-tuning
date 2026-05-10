# rag-pgvector-tuning

**What you run:** a small **[FastAPI](https://fastapi.tiangolo.com/)** service plus **[PostgreSQL](https://www.postgresql.org/)** with **[pgvector](https://github.com/pgvector/pgvector)**. You **ingest** text chunks as vectors, **retrieve** nearest neighbors for a query vector, and **tune** pgvector search knobs (mainly HNSW `ef_search` or IVFFlat `probes`) using YAML profiles, optional runtime overrides, and a demo **telemetry + tuner** loop.

**What this repo does *not* include:** prompt templates, chat orchestration, or a hosted LLM product. You bring your own model caller if you want answers—not just retrieved passages.

### Architecture (one path through the system)

1. **Config at startup** — `config/embedding.yaml` (vector dimension), `config/profiles.yaml` (HNSW vs IVFFlat defaults), `config/tuner_guardrails.yaml` (tuner bounds).
2. **Ingest** — `POST /ingest/chunks` turns each chunk’s text into an embedding via the configured **embedding backend**, then upserts rows into table `chunks`.
3. **Retrieve** — `POST /retrieve` embeds the query text with the **same** backend, runs a pgvector nearest-neighbor query under the **active profile’s** session parameters (`ef_search` / `probes`).
4. **Tune** — in-process telemetry records retrieve latency; `/tuner/recommend` suggests parameter moves within guardrails.

### Scope (this repository)

- **Retrieval stack:** PostgreSQL + pgvector, SQL migrations, ingest and k-NN retrieve, YAML-driven HNSW / IVFFlat query knobs, in-process telemetry and MVP tuner.
- **Embeddings:** pluggable backends ([see below](#embedding-backends)) — `demo` (hash), OpenAI-compatible HTTP, or local HTTP.
- **Operations:** optional strict `tenant_id` on retrieve, CORS allowlist, disabling `/docs`, optional per-IP rate limit (see [Operations](#operations-environment-variables)).
- **Metadata:** per-chunk JSON on ingest and `metadata_filter` on retrieve (Postgres `@>` containment).
- **Learning helpers:** optional [`scripts/eval_recall.py`](scripts/eval_recall.py) for a small recall@k smoke-style check.

For limits and threat assumptions, see [SECURITY.md](SECURITY.md). This repo does not ship a full LLM orchestration layer or a hosted offering — bring your own caller and deployment hardening.

### Embedding backends

| Value | Meaning |
| ----- | ------- |
| `demo` (default) | Deterministic hash-derived vectors—**no API keys**, good for latency demos; **not** semantic similarity. |
| `openai` | `POST {OPENAI_BASE_URL}/embeddings` — set **`OPENAI_API_KEY`**, match **`config/embedding.yaml`** dimension to the model (e.g. `text-embedding-3-small`). |
| `local` | Same HTTP shape as OpenAI; set **`LOCAL_EMBEDDINGS_BASE_URL`** (e.g. Ollama, TEI). Optional **`LOCAL_EMBEDDINGS_API_KEY`**. |

> [!CAUTION]
> **Attention — embedding dimension**
>
> **Misaligned** dimension — **`config/embedding.yaml`** and the **DB** must match your model’s **actual vector length** — or **ingest** and **retrieve** will **fail**. The API returns a **clear error** so you can fix the config or backend.

### Who this is for

Developers learning **RAG retrieval** and **pgvector** who want a **reproducible** HTTP API and SQL migrations—not a full production retrieval platform.

### What you need installed

- **Docker** and **Docker Compose v2** (`docker compose`). If your machine only has the legacy hyphenated binary, use **`docker-compose`** instead (same flags).
- **Python 3.11+** and **[uv](https://github.com/astral-sh/uv)** — for running the API on the host against Docker Postgres.

## Quick start

### Option A — Everything in Docker (Postgres + API)

From the project root:

```bash
docker compose up --build -d
```

Then follow **[Step-by-step test guide](#path-1-docker-full-stack-api--postgres)** (Path 1: Docker) below. Postgres is also on **host port 5433** (`postgresql://rag:rag@localhost:5433/rag`) if you use external tools.

### Option B — Postgres in Docker, API on your machine

1. **Start the database**
  ```bash
   docker compose up -d postgres
  ```
2. **Point the app at the database** (default matches `docker-compose.yml`)
  ```bash
   export DATABASE_URL=postgresql://rag:rag@localhost:5433/rag
  ```
   You can also copy [.env.example](.env.example) to `.env` and edit the URL (loaded by `src/rag/settings.py`).
3. **Install dependencies and run migrations**
  ```bash
   uv sync
   uv run python scripts/migrate.py
  ```
   For an **IVFFlat-oriented** profile and alternate SQL, see the section *Switching index family* below.
4. **Run the API**
  ```bash
   uv run uvicorn rag.main:app --host 0.0.0.0 --port 8000 --app-dir src
  ```
5. **Open interactive docs**
  [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
6. Continue with **[Step-by-step test guide](#path-2-local-api--docker-postgres-only)** (Path 2) from step 3 onward—use the same base URL `http://127.0.0.1:8000`.

## Step-by-step test guide

Work through the steps in order: **load vectors**, **query**, then **change search parameters** and compare timings.

**What you should see**

| Step area | What to notice |
| --------- | ---------------- |
| Ingest + retrieve | `results` rows include `doc_id`, `content`, `cosine_sim`, and optional **`metadata`**. Embedding dimension follows **`config/embedding.yaml`**. |
| Retrieve response | **`duration_ms`**, active **`profile`**, effective **`hnsw_ef_search`** or **`ivfflat_probes`**. |
| Active profile | Values from `config/profiles.yaml` plus optional runtime overrides. |
| Runtime PATCH | Lower **`ef_search`** often lowers **latency** (and may lower **recall**). With **`EMBEDDING_BACKEND=demo`**, vectors are not semantic—**latency is the clearest signal**. |
| Telemetry / tuner | Rolling latency percentiles; `/tuner/recommend` proposes bounded moves. |

If `docker compose` is missing, install **Compose V2** (Docker Desktop / plugin). Some older setups only provide **`docker-compose`**.

### Path 1: Docker full stack (API + Postgres)

1. **Start the stack** — Postgres + API; migrations run on API startup.
  ```bash
  docker compose up --build -d
  ```
2. **Confirm the API is healthy** — migrations must succeed before traffic is safe.
  ```bash
  docker compose ps
  docker compose logs -f api
  ```
   (`Ctrl+C` stops following logs; containers keep running.)
3. **Open Swagger UI** — try endpoints interactively: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
4. **Ingest two chunks** — embeddings use **`EMBEDDING_BACKEND`** (default `demo`). Rows upsert on **`(doc_id, chunk_index)`** (tenant isolation is optional filtering—see [SECURITY.md](SECURITY.md)).
  ```bash
  curl -s -X POST http://127.0.0.1:8000/ingest/chunks \
    -H "Content-Type: application/json" \
    -d '{"chunks":[{"tenant_id":"demo","source_type":"doc","doc_id":"doc-a","chunk_index":0,"content":"pgvector stores embeddings in PostgreSQL."},{"tenant_id":"demo","source_type":"doc","doc_id":"doc-b","chunk_index":0,"content":"HNSW ef_search trades latency for recall."}]}'
  ```
   Expect `{"upserted":2,...}`.
5. **Retrieve** — query embedding + k-NN search under the active profile. **`tenant_id`**, **`source_type`**, and **`metadata_filter`** narrow candidates **before** ordering by distance.
  ```bash
  curl -s -X POST http://127.0.0.1:8000/retrieve \
    -H "Content-Type: application/json" \
    -d '{"query":"What is ef_search?","k":5,"tenant_id":"demo"}'
  ```
   Record **`duration_ms`** and **`hnsw_ef_search`** for the next steps.
6. **Inspect active profile** (YAML + overrides):
  ```bash
   curl -s http://127.0.0.1:8000/config/active-profile
  ```
7. **Apply a runtime override** (must stay within `config/tuner_guardrails.yaml`). Example: lower `ef_search`:
  ```bash
   curl -s -X PATCH http://127.0.0.1:8000/config/runtime-search \
     -H "Content-Type: application/json" \
     -d '{"hnsw_ef_search": 24}'
  ```
8. **Retrieve again** with the **same body** as step 5. Compare **`duration_ms`** (and `hnsw_ef_search` should reflect `24` if within guardrails).
9. **Clear overrides** (back to profile defaults):
  ```bash
   curl -s -X PATCH http://127.0.0.1:8000/config/runtime-search \
     -H "Content-Type: application/json" \
     -d '{"clear_overrides": true}'
  ```
10. **Telemetry:** run a few extra retrieves from `/docs`, then:
  ```bash
    curl -s http://127.0.0.1:8000/telemetry/summary
  ```
11. **Tuner (optional):** with enough retrieve traffic, ask for a recommendation:
  ```bash
    curl -s -X POST http://127.0.0.1:8000/tuner/recommend
  ```
    Optional auto-apply step (understand guardrails first):
  ```bash
    curl -s -X POST "http://127.0.0.1:8000/tuner/step?auto_apply=false"
  ```
12. **Edit config without rebuilding:** change `config/profiles.yaml` or `config/tuner_guardrails.yaml`, then:
  ```bash
    docker compose restart api
  ```
    Re-run steps 5–6 to see new defaults.
13. **Tear down:**
  ```bash
    docker compose down
  ```

### Path 2: Local API + Docker Postgres only

1. Complete **Option B** in [Quick start](#quick-start) (Postgres on `localhost:5433`, API on port 8000).
2. Continue from **Path 1, step 3** above using [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and the same `curl` commands.
3. For YAML changes, **restart your local uvicorn** instead of `docker compose restart api`.

## Example: ingest chunks

Upserts into `chunks` (vector dimension from **`config/embedding.yaml`**; embedding values from **`EMBEDDING_BACKEND`**). Optional per-chunk **`metadata`** object is stored as JSONB for **`metadata_filter`** on retrieve.

```bash
curl -s -X POST http://127.0.0.1:8000/ingest/chunks \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {
        "tenant_id": "demo",
        "source_type": "doc",
        "doc_id": "manual-1",
        "chunk_index": 0,
        "content": "PostgreSQL can store vectors with the pgvector extension.",
        "metadata": {"section": "intro", "lang": "en"}
      }
    ]
  }'
```

Expected response shape: `{"upserted": <n>, "duration_ms": <float>}`.

## Example: retrieve similar chunks

```bash
curl -s -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I store embeddings?",
    "k": 5,
    "tenant_id": "demo"
  }'
```

The API uses the **active profile** from `config/profiles.yaml` to set session parameters (e.g. HNSW `ef_search`) before running the query. The response includes match rows, timings, and which profile was used.

Optional filters (applied in SQL **before** vector ordering):

- `tenant_id` — restrict to one tenant string
- `source_type` — e.g. only `"doc"` chunks
- `metadata_filter` — JSON object; rows must satisfy **`metadata @> metadata_filter`** (Postgres containment). Example:

```bash
curl -s -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"vectors","k":5,"tenant_id":"demo","metadata_filter":{"section":"intro"}}'
```

## Example: see active tuning profile

```bash
curl -s http://127.0.0.1:8000/config/active-profile
```

## Example: telemetry summary

```bash
curl -s http://127.0.0.1:8000/telemetry/summary
```

## Example: tuner (recommend and optional auto-apply)

The tuner reads **in-process** telemetry (latency percentiles). It does not replace a full production control plane; it is an **MVP** to show the idea.

Get a recommendation:

```bash
curl -s -X POST http://127.0.0.1:8000/tuner/recommend
```

Apply one step if the recommendation says so (`auto_apply=true`):

```bash
curl -s -X POST "http://127.0.0.1:8000/tuner/step?auto_apply=true"
```

Override search knobs for one session (allowed keys are **whitelisted** in `config/tuner_guardrails.yaml`):

```bash
curl -s -X PATCH http://127.0.0.1:8000/config/runtime-search \
  -H "Content-Type: application/json" \
  -d '{"hnsw_ef_search": 48}'
```

Clear overrides:

```bash
curl -s -X PATCH http://127.0.0.1:8000/config/runtime-search \
  -H "Content-Type: application/json" \
  -d '{"clear_overrides": true}'
```

## CLI (`rag-cli`)

After **`uv sync`**, use the **`rag-cli`** entrypoint for JSON endpoints:

```bash
uv run rag-cli ingest --file chunks.json
uv run rag-cli retrieve --query "What is ef_search?" --k 5 --tenant-id demo
uv run rag-cli retrieve --query "vectors" --k 5 --tenant-id demo --metadata-json '{"section":"intro"}'
uv run rag-cli tune-step --auto-apply
```

Same via **`python -m rag.cli …`**. **`--base-url`** defaults to `http://127.0.0.1:8000` or **`RAG_BASE_URL`**; **`RAG_API_KEY`** / **`API_KEY`** supply **`X-API-Key`** when **`--api-key`** is omitted.

See **`rag-cli --help`** and **`rag-cli <command> --help`**.

## Load generator (retrieve QPS)

Use [`scripts/load_retrieve_qps.py`](scripts/load_retrieve_qps.py) to issue paced **`POST /retrieve`** traffic so **`/telemetry/summary`** and **`/tuner/recommend`** see sustained latency samples (hash embeddings make latency an easy knob to observe).

```bash
uv sync
uv run python scripts/load_retrieve_qps.py --qps 15 --duration 45
```

With **`RAG_API_KEY`** enabled on the API:

```bash
uv run python scripts/load_retrieve_qps.py --api-key "$RAG_API_KEY" --qps 10 --duration 30 --tenant-id demo
```

Machine-readable summary:

```bash
uv run python scripts/load_retrieve_qps.py --qps 20 --duration 20 --json
```

See **`python scripts/load_retrieve_qps.py --help`** for options (`--base-url`, `--query`, `--k`, `--concurrency`, …).

## Minimal recall check

[`scripts/eval_recall.py`](scripts/eval_recall.py) ingests a tiny labeled corpus, runs **`POST /retrieve`** for fixed queries (including one **`metadata_filter`** case), and prints whether expected **`doc_id`** values appear in the top‑**k** results. With **`EMBEDDING_BACKEND=demo`**, rankings are **not** semantic—expect **`hit: false`** sometimes; use **`openai`** or **`local`** for meaningful recall.

```bash
uv run python scripts/eval_recall.py --k 5
```

Use **`--skip-ingest`** when the corpus is already loaded.

## Configuration files (read this before emailing anyone)


| File                           | Purpose                                                                                                                                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config/embedding.yaml`        | Declares **embedding dimension** and model label. Must match the output size of **`EMBEDDING_BACKEND`** (`openai` / `local`).                                                                             |
| `config/profiles.yaml`         | **Which index family** (HNSW vs IVFFlat), **build** hints, and **default search** parameters (`hnsw_ef_search`, `ivfflat_probes`). Set `active_profile` to the profile name you want.                   |
| `config/tuner_guardrails.yaml` | **Bounds** for the tuner: min/max `ef_search` and probes, max change per step, **cooldown** between auto changes, **target_p99_latency_ms**, rollback multipliers, and **whitelist** of runtime params. |


## Switching index family (HNSW vs IVFFlat)

- Default migrations create an **HNSW** index (see `migrations/003_index_hnsw.sql`).
- The profile `high_qps_approximate` in `config/profiles.yaml` is documented as **IVFFlat-oriented**. To align the database with that profile you must use the **alternate** migration path described in code comments and run:
  ```bash
  uv run python scripts/migrate.py --alternate-ivfflat
  ```
  (Only after you have followed any SQL/index swap steps your deployment needs; see `migrations/alternate/ivfflat_profile.sql`.)

Do not flip `active_profile` to IVFFlat in YAML unless your database actually has a matching **IVFFlat** index — otherwise searches will be wrong or fail.

## Operations (environment variables)

See [.env.example](.env.example) for a concise list. Highlights:

- **`GET /health`** — process liveness only (no database).
- **`GET /ready`** — DB connectivity, **pgvector** loaded, **`chunks`** exists (**503** if not).
- **`DATABASE_URL`** — async Postgres DSN for the API.
- **`MAX_INGEST_CHUNKS_PER_REQUEST`** — max chunks per ingest request (default **500**, **413** if exceeded).
- **`MAX_CHUNK_CONTENT_CHARS`**, **`MAX_RETRIEVE_QUERY_CHARS`** — body size limits (**422** when exceeded).
- **`EMBEDDING_BACKEND`**, **`OPENAI_*`**, **`LOCAL_EMBEDDINGS_*`**, **`EMBEDDING_HTTP_TIMEOUT_S`** — see [Embedding backends](#embedding-backends).
- **`REQUIRE_TENANT_ID`** — when `true`, **`POST /retrieve`** requires a non-empty **`tenant_id`** (**400** otherwise).
- **`CORS_ORIGINS`** — comma-separated browser origins; unset keeps framework defaults.
- **`DISABLE_OPENAPI_UI`** — when `true`, **`/docs`** and **`/redoc`** are disabled (use in untrusted networks).
- **`RATE_LIMIT_PER_MINUTE`** — optional **per-IP** limit for this single process (**429**); **not** a multi-worker/global quota—prefer a **reverse proxy** in production (see [SECURITY.md](SECURITY.md)).
- **`RAG_API_KEY`** / **`API_KEY`** — optional shared secret; when set, all routes except **`/health`** and **`/ready`** require **`Authorization: Bearer`** or **`X-API-Key`**.

In **`docker-compose.yml`**, the **`api`** service **`healthcheck`** uses **`GET /ready`** (`start_period` allows migrations).

## Run tests

```bash
uv sync --group dev   # or: uv sync --extra dev
uv run pytest tests/ -q --cov=rag --cov-branch
```

- **Unit and API tests** use a **mocked PostgreSQL pool** (no Docker required).
- **OpenAPI contract:** [`tests/fixtures/openapi.json`](tests/fixtures/openapi.json) must match `app.openapi()` (see [`tests/test_openapi_contract.py`](tests/test_openapi_contract.py)). After intentional route or schema changes, refresh the golden file with `UPDATE_OPENAPI_SNAPSHOT=1 uv run pytest tests/test_openapi_contract.py -q` and commit the updated JSON (minor drift after `fastapi` / `pydantic` upgrades is normal).
- **Integration tests** are marked `@pytest.mark.integration`: they connect to PostgreSQL (default `postgresql://rag:rag@localhost:5433/rag`). Start the DB with `docker compose up -d postgres`, or set `INTEGRATION_DATABASE_URL`. If the server is unreachable, the integration test is **skipped**.

### Continuous integration

[GitHub Actions](.github/workflows/ci.yml) runs **pytest with coverage** on every push and pull request; a **`security` job** runs **Bandit** on `src/rag` and `scripts` plus **blocking `pip-audit`** on exported locked runtime dependencies; and a separate job exercises **PostgreSQL with pgvector** for integration tests. [SECURITY.md](SECURITY.md) maps application risks to **OWASP Top 10:2025** and aligns with those checks.

## Project layout

- `Dockerfile` — API image (runs migrations then uvicorn)
- `docker-compose.yml` — Postgres (pgvector) + API
- `docker/entrypoint.sh` — migrate + start server
- `src/rag/` — FastAPI app, tuner, telemetry, config loading; [`cli.py`](src/rag/cli.py) provides **`rag-cli`**
- `migrations/` — numbered SQL files applied by `scripts/migrate.py`
- `scripts/migrate.py` — idempotent migration runner
- `scripts/load_retrieve_qps.py` — paced `/retrieve` load generator for tuner demos
- `scripts/eval_recall.py` — tiny recall@k harness (ingest + retrieve + metadata filter)
- `src/rag/embedding_backend.py` — pluggable embedding providers
- `tests/` — pytest (tuner, API with mocked DB, optional integration against real Postgres)

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).
