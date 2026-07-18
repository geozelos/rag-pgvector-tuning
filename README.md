# Tune pgvector RAG Latency in PostgreSQL

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/geozelos/rag-pgvector-tuning/actions/workflows/ci.yml/badge.svg)](https://github.com/geozelos/rag-pgvector-tuning/actions/workflows/ci.yml)
[![pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg)](https://github.com/pgvector/pgvector)

**See how HNSW `ef_search` trades latency for recall** — runnable FastAPI + Postgres, zero embedding API keys.

```text
ef_search → latency (p50) and recall@k vs oracle

    ef    p50_ms   recall  latency bar / recall bar
------------------------------------------------------------------------
     8     1.023    0.267  L|██░░░░░░░░░░░░░░░░░░| R|█████░░░░░░░░░░░░░░░|
    16     0.983    0.300  L|░░░░░░░░░░░░░░░░░░░░| R|██████░░░░░░░░░░░░░░|
    40     1.142    0.400  L|████████░░░░░░░░░░░░| R|████████░░░░░░░░░░░░|
    96     1.387    0.467  L|████████████████████| R|█████████░░░░░░░░░░░|
   200     1.346    0.467  L|██████████████████░░| R|█████████░░░░░░░░░░░|

L = p50 duration_ms (relative) · R = mean |hit∩oracle| / |oracle| over gold queries
```

> Measured chart (also in [`docs/assets/latency-recall-chart.txt`](docs/assets/latency-recall-chart.txt)). Reproduce: `make up && make demo`.

## First run (~5–10 min)

```bash
make up                        # or: make up COMPOSE=docker-compose
make demo                      # seed 10k + modest HNSW rebuild → chart
# → docs/assets/latency-recall-chart.txt (+ .png if Pillow available)
```

No Pinecone. No LangChain maze. No OpenAI key for the default path.

**Recall method:** exact cosine top-`k` (`exact=true`, seq scan) vs HNSW at each `ef_search`.
`make demo` rebuilds a modest HNSW (`m=4`, `ef_construction=8`) so the knob is visible on a 10k lab corpus.

**Problem:** RAG tutorials stop at “call the LLM.” Production pain is often **retrieval latency and recall** inside Postgres.

**Outcome:** Change `ef_search`, re-run retrieve, read `duration_ms` — and see recall move on a chart.

---

## Tuning walkthrough

1. **Ingest** (demo embeddings — feature hashing, topical similarity, no API keys):
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
4. **Retrieve again** — compare `duration_ms`.
5. Optional: `GET /telemetry/summary`, `POST /tuner/recommend`.

OpenAPI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## What you get

| You get | Why it matters |
| ------- | -------------- |
| **Inspectable stack** | FastAPI + PostgreSQL + pgvector — migrations, SQL, handlers |
| **Latency × recall demo** | `make demo` writes a shareable chart (ASCII + optional PNG) |
| **YAML search profiles** | HNSW / IVFFlat defaults, runtime overrides, guardrails |
| **Pluggable embeddings** | `demo` (offline feature-hash), or OpenAI-compatible / local HTTP |
| **Recipes** | [docs/](docs/README.md) — load testing, IVFFlat, hardening |

**Not included (on purpose):** chat UI, prompt templates, hosted LLM product.

### Embedding backends

| Value | Meaning |
| ----- | ------- |
| `demo` (default) | Offline feature-hash vectors — **no API keys**; topical overlap is preserved for demos |
| `openai` | Set `OPENAI_API_KEY`; match dimension in `config/embedding.yaml` |
| `local` | OpenAI-compatible HTTP (Ollama, TEI) via `LOCAL_EMBEDDINGS_BASE_URL` |

> [!CAUTION]
> **Embedding dimension** — `config/embedding.yaml` and the DB must match your model's vector length.

---

## Operator shortcuts

```bash
make help                  # list targets
make up / make down
make demo                  # seed + latency×recall chart
make benchmark-latency     # longer load-based latency sweep
make test / make integration / make security
```

**Local API + Docker Postgres:** [docs/recipes/local-api-docker-postgres.md](docs/recipes/local-api-docker-postgres.md)

**CLI:**

```bash
uv run rag-cli retrieve --query "What is ef_search?" --k 5 --tenant-id demo
uv run rag-cli tune-step --auto-apply
```

---

## Configuration

| File | Purpose |
| ---- | ------- |
| `config/embedding.yaml` | Dimension + model label |
| `config/profiles.yaml` | Index family + default search params |
| `config/tuner_guardrails.yaml` | Bounds, cooldown, whitelist |

Env highlights: [.env.example](.env.example). Threat model: [SECURITY.md](SECURITY.md).

## Tests

```bash
uv sync --group dev
uv run pytest tests/ -q --cov=rag --cov-branch
```

## License

MIT — see [LICENSE](LICENSE).
