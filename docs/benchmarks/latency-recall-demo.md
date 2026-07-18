# ef_search latency × recall demo

Offline **feature-hash** embeddings + topical corpus — no API keys.

**Recall:** mean fraction of exact (seq-scan) top-`k` `doc_id`s recovered at each approximate `ef_search`.

## Setup

| Parameter | Value |
| --- | --- |
| Date (UTC) | 2026-07-18 11:39:42 UTC |
| API | `http://127.0.0.1:8000` |
| Tenant | `demo` |
| k | 10 |
| Oracle | `exact_seqscan_topk_doc_id` |
| Samples / ef | 25 (+ 3 warmup) |
| Corpus chunks | 10000 |

## Chart

```
ef_search → latency (p50) and recall@k vs oracle

    ef    p50_ms   recall  latency bar / recall bar
------------------------------------------------------------------------
     8     0.842    0.067  L|░░░░░░░░░░░░░░░░░░░░| R|█░░░░░░░░░░░░░░░░░░░|
    16     0.852    0.067  L|█░░░░░░░░░░░░░░░░░░░| R|█░░░░░░░░░░░░░░░░░░░|
    40     0.877    0.067  L|██░░░░░░░░░░░░░░░░░░| R|█░░░░░░░░░░░░░░░░░░░|
    96     0.894    0.067  L|████░░░░░░░░░░░░░░░░| R|█░░░░░░░░░░░░░░░░░░░|
   200     1.127    0.533  L|████████████████████| R|███████████░░░░░░░░░|

L = p50 duration_ms (relative) · R = mean |hit∩oracle| / |oracle| over gold queries
```

## Table

| `hnsw_ef_search` | p50 (ms) | p99 (ms) | recall@k |
| --- | ---: | ---: | ---: |
| 8 | 0.842 | 1.212 | 0.067 |
| 16 | 0.852 | 1.046 | 0.067 |
| 40 | 0.877 | 1.354 | 0.067 |
| 96 | 0.894 | 1.252 | 0.067 |
| 200 | 1.127 | 1.31 | 0.533 |

> Hardware-specific. Reproduce: `make up && make demo` (use `COMPOSE=docker-compose` if needed).
