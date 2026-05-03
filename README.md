# rag-pgvector-tuning

**RAG (retrieval-augmented generation) data**, **PostgreSQL with [pgvector](https://github.com/pgvector/pgvector)**, and **runtime tuning** of vector search parameters.

## What this project is (in plain language)

- **RAG** means: you store text **chunks** with **embedding vectors** (numbers that represent meaning), then **search** for chunks similar to a user question to feed an LLM. This repo focuses on the **retrieval** and **database** side, not on calling OpenAI or similar APIs.
- **pgvector** is a PostgreSQL extension that stores vectors and finds nearest neighbors (similar vectors) quickly.
- **Physical tuning** here means: adjusting **query-time** settings for the index, mainly **HNSW `ef_search`** or **IVFFlat `probes`**, to trade **speed vs. recall/quality**. The API can suggest changes from simple telemetry and apply them within **guardrails** you configure in YAML.

Embeddings in this demo are **deterministic fake vectors** (hash-based). That keeps the repo runnable **without API keys** while you learn ingestion, search, and tuning.

## Who this is for

Developers new to RAG or pgvector who want a **small, runnable FastAPI service** with:

- Docker Postgres + pgvector
- SQL migrations
- HTTP endpoints for ingest, retrieve, and tuning
- Config files you can edit without touching Python

## What you need installed

- **Docker** and **Docker Compose** v2 (`docker compose`) — for Option A (full stack) or Option B (Postgres only)
- **Python 3.11+** and **[uv](https://github.com/astral-sh/uv)** — for Option B (API on the host)

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

Use this checklist to **see the project working end-to-end**: data in pgvector, similarity search, YAML profiles, runtime overrides, and optional telemetry/tuner.

**What you should get out of it**


| Step area         | What to notice                                                                                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ingest + retrieve | JSON with `results`; vectors are **768-d** (see `config/embedding.yaml`)                                                                                                                    |
| Retrieve response | `duration_ms`, `hnsw_ef_search` / profile name                                                                                                                                              |
| Active profile    | Values from `config/profiles.yaml` plus any overrides                                                                                                                                       |
| Runtime PATCH     | After changing `hnsw_ef_search`, **compare** `duration_ms` on the same query (lower `ef_search` often lowers latency; this demo uses hash embeddings so **latency is the clearest signal**) |
| Telemetry / tuner | Rolling stats; MVP suggestions from `/tuner/recommend`                                                                                                                                      |


If `docker compose` is not found, try **Docker Compose V2** via Docker Desktop or the plugin command `docker compose` (with a space). Older installs may use `docker-compose`.

### Path 1: Docker full stack (API + Postgres)

1. **Prerequisites:** Docker with Compose. From the repo root, start the stack:
  ```bash
   docker compose up --build -d
  ```
2. **Verify containers:** API should start after migrations complete.
  ```bash
   docker compose ps
   docker compose logs -f api
  ```
   (`Ctrl+C` leaves containers running; only stops log follow.)
3. **Open Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
4. **Ingest two chunks** (or use “Try it out” on `POST /ingest/chunks`):
  ```bash
   curl -s -X POST http://127.0.0.1:8000/ingest/chunks \
     -H "Content-Type: application/json" \
     -d '{"chunks":[{"tenant_id":"demo","source_type":"doc","doc_id":"doc-a","chunk_index":0,"content":"pgvector stores embeddings in PostgreSQL."},{"tenant_id":"demo","source_type":"doc","doc_id":"doc-b","chunk_index":0,"content":"HNSW ef_search trades latency for recall."}]}'
  ```
   Expect `{"upserted":2,...}`.
5. **Retrieve** and note **timings and search knob** in the response:
  ```bash
   curl -s -X POST http://127.0.0.1:8000/retrieve \
     -H "Content-Type: application/json" \
     -d '{"query":"What is ef_search?","k":5,"tenant_id":"demo"}'
  ```
   Save or remember **`duration_ms`** and **`hnsw_ef_search`** for comparison later.
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

This inserts (or updates) rows in the `chunks` table. Vectors are **768-dimensional** in the default config (`config/embedding.yaml`).

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
        "content": "PostgreSQL can store vectors with the pgvector extension."
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

Optional filters:

- `tenant_id` — only chunks for that tenant
- `source_type` — e.g. only `"doc"` chunks

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

## Configuration files (read this before emailing anyone)


| File                           | Purpose                                                                                                                                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config/embedding.yaml`        | Declares **embedding dimension** and a label for the demo “model”. Replace the demo embedder in code when you use a real model.                                                                         |
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

## Operations (readiness, ingest limits, optional API key)

- **`GET /health`** — process liveness only (no database call). Use for “is the Python process up?” probes.
- **`GET /ready`** — readiness after migrations: PostgreSQL connectivity, **pgvector** extension loaded, and the **`chunks`** table exists. HTTP **503** with a JSON body when something is missing or the DB is unreachable — suitable for orchestrators that should not send traffic until the DB is usable.
- **`MAX_INGEST_CHUNKS_PER_REQUEST`** — caps how many items you may send in one `POST /ingest/chunks` body (default **500**). Larger batches receive HTTP **413**.
- **`RAG_API_KEY`** — optional shared secret. If set in `.env`, every route except **`GET /health`** and **`GET /ready`** requires `Authorization: Bearer <key>` or `X-API-Key: <key>`. The same value may be supplied via **`API_KEY`** (pydantic-settings alias). Leave unset for local demos. See **[SECURITY.md](SECURITY.md)** for production posture beyond this stub.

In **`docker-compose.yml`**, the **`api`** service defines a **`healthcheck`** against **`GET /ready`** so Compose can treat the container as healthy only after migrations and DB checks succeed (`start_period` allows time for first-boot migrations).

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
- `scripts/load_retrieve_qps.py` — paced `/retrieve` load generator for tuner demos (**`uv sync`** includes **httpx**; use **`uv sync --group dev`** when running pytest alongside)
- `tests/` — pytest (tuner, API with mocked DB, optional integration against real Postgres)

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).
