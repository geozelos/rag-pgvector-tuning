# Recipe: Benchmark ef_search latency

**Goal:** Produce reproducible **HNSW `ef_search` vs latency** numbers for the README and [`docs/benchmarks/latency-ef-search.md`](../benchmarks/latency-ef-search.md).

## Prerequisites

- Running API (`make up` or local uvicorn + Docker Postgres)
- Default **`EMBEDDING_BACKEND=demo`** (latency signal without API keys)

## Steps

```bash
make up
make seed-demo
make benchmark-latency
```

Artifacts:

- `docs/benchmarks/results.json` — machine-readable samples
- `docs/benchmarks/latency-ef-search.md` — methodology + markdown table

## README assets (optional)

Regenerate terminal-style PNGs for the README demo section:

```bash
uv run --with pillow python scripts/generate_readme_assets.py
```

Writes `docs/assets/retrieve-duration.png` and `docs/assets/ef-search-comparison.png`.

## Customize

```bash
uv run python scripts/benchmark_ef_search.py \
  --ef-values 16,32,64,128 \
  --qps 20 \
  --duration 30 \
  --tenant-id demo
```

## Note

The benchmark records API **`duration_ms`** from `/retrieve` JSON responses (p50/p99 under paced load), not raw HTTP round-trip time. Requires at least `--corpus-chunks` rows in the DB — run **`make seed-demo`** first.

With **`demo`** embeddings (offline feature hashing), topical overlap is preserved for lab demos.
For a latency × recall chart prefer `make demo`. For a tiny labeled smoke check see [eval-recall.md](eval-recall.md).
