# HNSW ef_search latency benchmark

Reproducible retrieve latency samples for pgvector **HNSW** query-time tuning.

## Setup

| Parameter | Value |
| --- | --- |
| Date (UTC) | 2026-06-08 19:52:51 UTC |
| API | `http://127.0.0.1:8000` |
| Embedding backend | `demo` |
| Corpus chunks | 400 (observed: 405) |
| Latency metric | api_duration_ms — p50/p99 of `/retrieve` response `duration_ms` under load |
| Index | HNSW (default migration) |
| Tenant | `demo` |
| Query | `HNSW ef_search latency benchmark` |
| k | 10 |
| Load | 15.0 req/s for 20.0s per knob (warmup 5.0s) |

## Results

| `hnsw_ef_search` | p50 (ms) | p99 (ms) | Notes |
| --- | ---: | ---: | --- |
| 16 | 5.265 | 8.136 | — |
| 24 | 5.924 | 9.515 | — |
| 40 | 5.933 | 12.532 | Default profile |
| 64 | 5.24 | 9.054 | Lowest p50 this run |
| 96 | 6.134 | 9.482 | Highest p50 this run |

> **Disclaimer:** Numbers depend on hardware, corpus size, and concurrent load.
> At ~400 demo chunks, `ef_search` spread is often small — increase corpus size for
> clearer separation. With `EMBEDDING_BACKEND=demo`, rankings are not semantic;
> this table shows **latency knobs**, not recall quality. For recall checks see
> [`scripts/eval_recall.py`](../../scripts/eval_recall.py).

## Reproduce

```bash
make up
make seed-demo
make benchmark-latency
```

Or directly:

```bash
uv run python scripts/seed_demo_corpus.py --chunks 400 --tenant-id demo
uv run python scripts/benchmark_ef_search.py --tenant-id demo
```
