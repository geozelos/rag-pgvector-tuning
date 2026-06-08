# Tune pgvector RAG Latency in PostgreSQL

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-green.svg)](.github/workflows/ci.yml)
[![pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg)](https://github.com/pgvector/pgvector)

> Measure how **HNSW `ef_search`** and **IVFFlat `probes`** change retrieval latency — with a runnable FastAPI stack, YAML profiles, and an MVP tuner loop.

**Problem:** RAG tutorials stop at “call the LLM.” Production pain is often **retrieval latency and recall** inside Postgres — knobs most teams guess at.

**Outcome:** In ~15 minutes, change `ef_search`, re-run retrieve, and read `duration_ms` + p99 telemetry. No Pinecone, no LangChain maze.

Keywords: **pgvector tuning**, **HNSW ef_search**, **IVFFlat probes**, **RAG performance**, **Postgres vector search**, **retrieval latency**.

---

## See it in 60 seconds

![POST /retrieve response with duration_ms and hnsw_ef_search](docs/assets/retrieve-duration.png)

```bash
docker compose up --build -d   # or: docker-compose up --build -d
make seed-demo                   # optional: pre-load demo corpus for richer results
curl -s -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"What is ef_search?","k":5,"tenant_id":"demo"}'
# → read duration_ms, profile, hnsw_ef_search in the JSON response
```

Lower `ef_search` and retrieve again — compare `duration_ms`:

![ef_search 24 vs 96 retrieve comparison](docs/assets/ef-search-comparison.png)

---

## Benchmark: `ef_search` vs latency

Measured on this repo with **400 demo chunks**, **HNSW** index, **15 req/s** load for 20s per knob. Percentiles are API **`duration_ms`** from `/retrieve` JSON responses ([full methodology](docs/benchmarks/latency-ef-search.md)).

| `hnsw_ef_search` | p50 (ms) | p99 (ms) | Notes |
| --- | ---: | ---: | --- |
| 16 | 5.265 | 8.136 | — |
| 24 | 5.924 | 9.515 | — |
| 40 | 5.933 | 12.532 | Default profile |
| 64 | 5.24 | 9.054 | Lowest p50 this run |
| 96 | 6.134 | 9.482 | Highest p50 this run |

> **Disclaimer:** Numbers depend on hardware, corpus size, and concurrent load. At ~400 demo chunks, `ef_search` spread is often small — seed a larger corpus for clearer separation. With `EMBEDDING_BACKEND=demo`, rankings are not semantic — this table shows **latency knobs**, not recall quality. Reproduce: `make up && make seed-demo && make benchmark-latency`.

---

## For who?

- **AI / ML engineers** — debug retrieval latency, not prompts
- **Postgres users** — pgvector as your vector search layer, no second vector DB
- **SaaS builders** — reference fork for ingest/retrieve, tenant scoping, metadata filters
- **Local LLM setups** — optional `local` embeddings (Ollama, TEI); latency tuning is the main story

### A good fit if you…

- Are learning **RAG retrieval** and want something **reproducible** (Docker, `uv`, OpenAPI)
- Need a **reference fork**, not a productized platform
- Like seeing **`ef_search` ↔ latency** with your own eyes before tuning production

### Probably not (yet) if you…

- Want a one-click chatbot — grab a higher-level toolkit; come back when retrieval feels fuzzy
- Need enterprise auth, quotas, and SLAs out of the box — harden using [SECURITY.md](SECURITY.md) and your own ops

---

## Quick start

```bash
docker compose up --build -d          # Postgres + API (host :8000, DB :5433)
curl -s http://127.0.0.1:8000/ready  # wait for 200
open http://127.0.0.1:8000/docs      # Swagger UI
```

**Local API + Docker Postgres only:** see [Option B](#option-b--postgres-in-docker-api-on-your-machine) below or [docs/recipes/local-api-docker-postgres.md](docs/recipes/local-api-docker-postgres.md).

---

## Tuning walkthrough (5 steps)

1. **Ingest** two chunks (demo embeddings, no API keys):
   ```bash
   curl -s -X POST http://127.0.0.1:8000/ingest/chunks \
     -H "Content-Type: application/json" \
     -d '{"chunks":[{"tenant_id":"demo","source_type":"doc","doc_id":"doc-a","chunk_index":0,"content":"pgvector stores embeddings in PostgreSQL."},{"tenant_id":"demo","source_type":"doc","doc_id":"doc-b","chunk_index":0,"content":"HNSW ef_search trades latency for recall."}]}'
   ```
2. **Retrieve** — note `duration_ms` and `hnsw_ef_search`:
   ```bash
   curl -s -X POST http://127.0.0.1:8000/retrieve \
     -H "Content-Type: application/json" \
     -d '{"query":"What is ef_search?","k":5,"tenant_id":"demo"}'
   ```
3. **Override** `ef_search` (within [config/tuner_guardrails.yaml](config/tuner_guardrails.yaml)):
   ```bash
   curl -s -X PATCH http://127.0.0.1:8000/config/runtime-search \
     -H "Content-Type: application/json" -d '{"hnsw_ef_search": 24}'
   ```
4. **Retrieve again** with the same body — compare `duration_ms`.
5. **Telemetry / tuner** (optional): run a few more retrieves, then `GET /telemetry/summary` and `POST /tuner/recommend`.

Full 13-step guide with IVFFlat notes: [Step-by-step test guide](#step-by-step-test-guide).

---

## What you get (and don't)

| You get | Why it matters |
| ------- | -------------- |
| **Inspectable stack** | [FastAPI](https://fastapi.tiangolo.com/) + [PostgreSQL](https://www.postgresql.org/) + [pgvector](https://github.com/pgvector/pgvector) — migrations, SQL, handlers you can read in an afternoon |
| **YAML-driven search profiles** | Swap HNSW vs IVFFlat defaults, patch runtime overrides, stay inside guardrails |
| **Pluggable embeddings** | `demo` (no API keys), or **OpenAI-compatible** / **local HTTP** for real similarity |
| **Metadata filters** | JSON on ingest, `@>` containment on retrieve — filters apply *before* vector ordering |
| **Recipes & scripts** | [Cookbooks in `docs/`](docs/README.md), [`rag-cli`](#cli-rag-cli), load generator, recall harness, [benchmarks](docs/benchmarks/latency-ef-search.md) |

**What you don't get (on purpose):** prompt templates, chat orchestration, or a hosted LLM product. You bring your own model caller for **answers** — this repo hands you **passages** and teaches you the **search**.

**Compared to…**

- **pgvector docs alone** — runnable API, migrations, repeatable `curl` experiments
- **Large RAG frameworks** — fewer abstractions; “here is the SQL and the session knob”
- **Managed vector DBs** — you operate Postgres yourself and learn what hosted services hide

---

## Architecture

1. **Config at startup** — `config/embedding.yaml`, `config/profiles.yaml`, `config/tuner_guardrails.yaml`
2. **Ingest** — `POST /ingest/chunks` embeds text and upserts into `chunks`
3. **Retrieve** — `POST /retrieve` embeds the query, runs k-NN under the active profile's session params
4. **Tune** — telemetry records latency; `/tuner/recommend` suggests bounded parameter moves

Threat model: [SECURITY.md](SECURITY.md).

### Embedding backends

| Value | Meaning |
| ----- | ------- |
| `demo` (default) | Hash-derived vectors — **no API keys**, good for latency demos; **not** semantic similarity |
| `openai` | `POST {OPENAI_BASE_URL}/embeddings` — set **`OPENAI_API_KEY`**, match dimension in **`config/embedding.yaml`** |
| `local` | Same HTTP shape; set **`LOCAL_EMBEDDINGS_BASE_URL`** (Ollama, TEI). Optional **`LOCAL_EMBEDDINGS_API_KEY`** |

> [!CAUTION]
> **Embedding dimension** — `config/embedding.yaml` and the DB must match your model's vector length or ingest/retrieve will fail.

---

## Operator ergonomics

- **[Makefile](Makefile):** `make help` — `up`, `down`, `migrate`, `seed-demo`, `benchmark-latency`, `test`, `integration`, `security`
- **Dev container:** [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) — Python + uv + Docker-from-Docker
- **Recipes:** [docs/README.md](docs/README.md) — embeddings, IVFFlat, load testing, hardening, troubleshooting

---

## Option B — Postgres in Docker, API on your machine

1. `docker compose up -d postgres` (or `docker-compose …`)
2. `export DATABASE_URL=postgresql://rag:rag@localhost:5433/rag`
3. `uv sync && uv run python scripts/migrate.py`
4. `uv run uvicorn rag.main:app --host 0.0.0.0 --port 8000 --app-dir src`
5. Continue from the [Tuning walkthrough](#tuning-walkthrough-5-steps)

---

## Step-by-step test guide

Extended walkthrough (Docker full stack + local API paths). If the table below happens, the project did its job.

**What you should see**

| Step area | What to notice |
| --------- | ---------------- |
| Ingest + retrieve | `results` rows include `doc_id`, `content`, `cosine_sim`, optional **`metadata`** |
| Retrieve response | **`duration_ms`**, active **`profile`**, effective **`hnsw_ef_search`** or **`ivfflat_probes`** |
| Runtime PATCH | Lower **`ef_search`** often lowers **latency** (may lower **recall**). With **`demo`**, latency is the clearest signal |
| Telemetry / tuner | Rolling percentiles; `/tuner/recommend` proposes bounded moves |

### Path 1: Docker full stack (API + Postgres)

1. `docker compose up --build -d`
2. `docker compose ps` and `docker compose logs -f api` until `/ready` returns 200
3. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
4. Ingest two chunks (see [Tuning walkthrough](#tuning-walkthrough-5-steps))
5. Retrieve — record **`duration_ms`** and **`hnsw_ef_search`**
6. `curl -s http://127.0.0.1:8000/config/active-profile`
7. PATCH runtime search (`hnsw_ef_search: 24`)
8. Retrieve again — compare **`duration_ms`**
9. Clear overrides: `{"clear_overrides": true}`
10. `curl -s http://127.0.0.1:8000/telemetry/summary`
11. `curl -s -X POST http://127.0.0.1:8000/tuner/recommend`
12. Edit `config/profiles.yaml`, then `docker compose restart api`
13. `docker compose down`

### Path 2: Local API + Docker Postgres only

Complete [Option B](#option-b--postgres-in-docker-api-on-your-machine), then follow Path 1 from step 3. Restart local uvicorn instead of `docker compose restart api` after YAML edits.

---

## Examples

### Ingest chunks

```bash
curl -s -X POST http://127.0.0.1:8000/ingest/chunks \
  -H "Content-Type: application/json" \
  -d '{"chunks":[{"tenant_id":"demo","source_type":"doc","doc_id":"manual-1","chunk_index":0,"content":"PostgreSQL can store vectors with pgvector.","metadata":{"section":"intro","lang":"en"}}]}'
```

### Retrieve with metadata filter

```bash
curl -s -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"vectors","k":5,"tenant_id":"demo","metadata_filter":{"section":"intro"}}'
```

### Active profile, telemetry, tuner

```bash
curl -s http://127.0.0.1:8000/config/active-profile
curl -s http://127.0.0.1:8000/telemetry/summary
curl -s -X POST http://127.0.0.1:8000/tuner/recommend
curl -s -X PATCH http://127.0.0.1:8000/config/runtime-search \
  -H "Content-Type: application/json" -d '{"hnsw_ef_search": 48}'
```

---

## CLI (`rag-cli`)

```bash
uv run rag-cli ingest --file chunks.json
uv run rag-cli retrieve --query "What is ef_search?" --k 5 --tenant-id demo
uv run rag-cli tune-step --auto-apply
```

`--base-url` defaults to `http://127.0.0.1:8000` or **`RAG_BASE_URL`**. See **`rag-cli --help`**.

---

## Load generator and recall

```bash
uv run python scripts/load_retrieve_qps.py --qps 15 --duration 45 --tenant-id demo
uv run python scripts/eval_recall.py --k 5
```

Benchmark corpus + ef_search sweep:

```bash
make seed-demo
make benchmark-latency
```

---

## Configuration files

| File | Purpose |
| ---- | ------- |
| `config/embedding.yaml` | Embedding **dimension** and model label |
| `config/profiles.yaml` | Index family (HNSW / IVFFlat), build hints, default search params |
| `config/tuner_guardrails.yaml` | Tuner bounds, cooldown, `target_p99_latency_ms`, runtime param whitelist |

## Switching index family (HNSW vs IVFFlat)

Default migrations create **HNSW** (`migrations/003_index_hnsw.sql`). For IVFFlat:

```bash
uv run python scripts/migrate.py --alternate-ivfflat
```

See [docs/recipes/ivfflat-profile.md](docs/recipes/ivfflat-profile.md). Do not set `active_profile` to IVFFlat unless the DB has a matching index.

## Operations (environment variables)

See [.env.example](.env.example). Highlights: `DATABASE_URL`, `EMBEDDING_BACKEND`, `REQUIRE_TENANT_ID`, `CORS_ORIGINS`, `RATE_LIMIT_PER_MINUTE`, `RAG_API_KEY` / `API_KEY`. Full checklist: [docs/recipes/api-hardening.md](docs/recipes/api-hardening.md).

## Run tests

```bash
uv sync --group dev
uv run pytest tests/ -q --cov=rag --cov-branch
```

[GitHub Actions](.github/workflows/ci.yml) runs pytest with coverage, Bandit + pip-audit, and PostgreSQL integration tests.

## Project layout

- `docs/` — recipes, [benchmarks](docs/benchmarks/latency-ef-search.md), [assets](docs/assets/)
- `src/rag/` — FastAPI app, tuner, telemetry, `rag-cli`
- `scripts/` — migrate, load generator, eval recall, seed corpus, benchmark ef_search
- `migrations/` — SQL applied by `scripts/migrate.py`
- `config/` — embedding dimension, profiles, tuner guardrails

## License

MIT — see [LICENSE](LICENSE).
