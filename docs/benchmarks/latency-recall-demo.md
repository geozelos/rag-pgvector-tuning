# ef_search latency × recall demo

Offline **feature-hash** embeddings + topical corpus — no API keys.

**Recall:** mean fraction of exact (seq-scan) top-`k` `doc_id`s recovered at each approximate `ef_search`.

## Setup

| Parameter | Value |
| --- | --- |
| Date (UTC) | 2026-07-18 11:35:54 UTC |
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
     8     1.023    0.267  L|██░░░░░░░░░░░░░░░░░░| R|█████░░░░░░░░░░░░░░░|
    16     0.983    0.300  L|░░░░░░░░░░░░░░░░░░░░| R|██████░░░░░░░░░░░░░░|
    40     1.142    0.400  L|████████░░░░░░░░░░░░| R|████████░░░░░░░░░░░░|
    96     1.387    0.467  L|████████████████████| R|█████████░░░░░░░░░░░|
   200     1.346    0.467  L|██████████████████░░| R|█████████░░░░░░░░░░░|

L = p50 duration_ms (relative) · R = mean |hit∩oracle| / |oracle| over gold queries
```

## Table

| `hnsw_ef_search` | p50 (ms) | p99 (ms) | recall@k |
| --- | ---: | ---: | ---: |
| 8 | 1.023 | 1.573 | 0.267 |
| 16 | 0.983 | 1.367 | 0.3 |
| 40 | 1.142 | 1.413 | 0.4 |
| 96 | 1.387 | 1.897 | 0.467 |
| 200 | 1.346 | 2.353 | 0.467 |

> Hardware-specific. Reproduce: `make up && make demo` (use `COMPOSE=docker-compose` if needed).
