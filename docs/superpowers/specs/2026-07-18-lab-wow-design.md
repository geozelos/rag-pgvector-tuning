# Lab-Wow Design — Latency × Recall Demo

**Status:** Approved by execution request (2026-07-18)  
**Goal:** Give GitHub visitors a zero-key, shareable proof that `ef_search` trades latency for recall.

## Problem

Default demo embeddings are hash-noise; the committed latency table is nearly flat. Visitors see no tradeoff → no stars.

## Approach

1. **Offline bag-of-words embeddings** (signed feature hashing) so topical text clusters without API keys.
2. **Larger seed corpus** with distinct topic vocabulary.
3. **`make demo`** runs stack → seed → sweep → ASCII (+ optional PNG) chart of p50 latency and recall@k vs `ef_search`.
4. **README** leads with that chart and a single CTA; real CI badge.

## Non-goals

Hybrid search, EXPLAIN UI, Pro features, OpenAI requirement.

## Success criteria

- `make demo` works with Docker only (no embedding API keys).
- Chart shows visible separation across `ef_search` values (recall and/or latency).
- Unit tests cover embedding similarity and demo-script dry-run.
