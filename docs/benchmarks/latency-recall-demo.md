# ef_search latency × recall demo

Offline **feature-hash** embeddings + topical corpus — no API keys.

**Recall:** mean fraction of exact (seq-scan) top-`k` row ids recovered at each approximate `ef_search`.

Each sweep point sends `hnsw_ef_search` on that `POST /retrieve` only. It does not change process-wide tuner state.

## Setup

| Parameter | Value |
| --- | --- |
| Date (UTC) | 2026-10-03 16:54:15 UTC |
| API | `http://127.0.0.1:8000` |
| Tenant | `demo` |
| k | 20 |
| HNSW build | m=6, ef_construction=16 |
| Oracle | `exact_seqscan_topk_doc_id` |
| Samples / ef | 25 (+ 3 warmup) |
| Corpus chunks | 10000 |

## Chart

```
ef_search → latency (p50) and recall@k vs oracle

    ef    p50_ms   recall  latency bar / recall bar
------------------------------------------------------------------------
     8     0.826    0.400  L|░░░░░░░░░░░░░░░░░░░░| R|████████░░░░░░░░░░░░|
    16     0.994    0.740  L|██████░░░░░░░░░░░░░░| R|███████████████░░░░░|
    40     1.129    0.900  L|██████████░░░░░░░░░░| R|██████████████████░░|
    96     1.433    0.980  L|████████████████████| R|████████████████████|
   200     1.122    0.980  L|██████████░░░░░░░░░░| R|████████████████████|

L = p50 duration_ms (relative) · R = mean |hit∩oracle| / |oracle| over gold queries
```

## Table

| `hnsw_ef_search` | p50 (ms) | p99 (ms) | recall@k |
| --- | ---: | ---: | ---: |
| 8 | 0.826 | 0.989 | 0.4 |
| 16 | 0.994 | 1.509 | 0.74 |
| 40 | 1.129 | 2.7 | 0.9 |
| 96 | 1.433 | 2.926 | 0.98 |
| 200 | 1.122 | 2.12 | 0.98 |

> Hardware-specific. Reproduce: `make up && make demo` (use `COMPOSE=docker-compose` if needed).
