#!/usr/bin/env python3
"""Tiny retrieval sanity check: labeled queries vs ``doc_id`` hits (recall@k).

Requires a running API (``uvicorn``). Uses ``POST /ingest/chunks`` then ``POST /retrieve``.

**Demo embeddings** (``EMBEDDING_BACKEND=demo``): vectors are hash-derived, so ranked order is **not**
semantic—recall numbers only sanity-check plumbing (filters, HTTP, pgvector wiring).

With **real embeddings** (OpenAI / local HTTP), recall@k measures whether expected documents appear in
the top-``k`` rows for each query (cheap offline-ish evaluation; not a full benchmark product).

Example::

    export RAG_BASE_URL=http://127.0.0.1:8000
    uv run python scripts/eval_recall.py --k 5

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx


def _recall_at_k(results: list[dict[str, Any]], expected: list[str], k: int) -> bool:
    doc_ids = [r.get("doc_id") for r in results[:k]]
    return any(d in doc_ids for d in expected)


def main() -> int:
    p = argparse.ArgumentParser(description="Minimal recall@k check via HTTP API.")
    p.add_argument(
        "--base-url",
        default=os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8000"),
        help="API origin (default: RAG_BASE_URL or http://127.0.0.1:8000)",
    )
    p.add_argument("--k", type=int, default=5, help="Top-k for retrieve and scoring")
    p.add_argument("--tenant-id", default="eval-recall", help="Tenant for ingest/retrieve")
    p.add_argument(
        "--api-key",
        default=os.environ.get("RAG_API_KEY") or os.environ.get("API_KEY"),
        help="Optional X-API-Key / Bearer",
    )
    p.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Only run retrieve (corpus must already contain expected doc_ids)",
    )
    args = p.parse_args()

    headers: dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    chunks = [
        {
            "tenant_id": args.tenant_id,
            "source_type": "eval",
            "doc_id": "eval-alpha",
            "chunk_index": 0,
            "content": "PostgreSQL pgvector stores embedding vectors for similarity search.",
            "metadata": {"topic": "vector-db"},
        },
        {
            "tenant_id": args.tenant_id,
            "source_type": "eval",
            "doc_id": "eval-beta",
            "chunk_index": 0,
            "content": "YAML profiles tune HNSW ef_search and IVFFlat probes at query time.",
            "metadata": {"topic": "tuning"},
        },
    ]

    queries: list[dict[str, Any]] = [
        {
            "query": "Where are embeddings stored?",
            "expect_docs": ["eval-alpha"],
            "metadata_filter": None,
        },
        {
            "query": "runtime search parameters",
            "expect_docs": ["eval-beta"],
            "metadata_filter": {"topic": "tuning"},
        },
    ]

    base = args.base_url.rstrip("/")
    with httpx.Client(base_url=base, headers=headers, timeout=120.0) as client:
        if not args.skip_ingest:
            ir = client.post("/ingest/chunks", json={"chunks": chunks})
            if ir.status_code >= 400:
                print(json.dumps({"error": "ingest_failed", "detail": ir.text}, indent=2))
                return 1

        hits = 0
        details: list[dict[str, Any]] = []
        for q in queries:
            body: dict[str, Any] = {
                "query": q["query"],
                "k": args.k,
                "tenant_id": args.tenant_id,
            }
            if q.get("metadata_filter") is not None:
                body["metadata_filter"] = q["metadata_filter"]
            rr = client.post("/retrieve", json=body)
            if rr.status_code >= 400:
                print(json.dumps({"error": "retrieve_failed", "detail": rr.text}, indent=2))
                return 1
            data = rr.json()
            results = data.get("results") or []
            ok = _recall_at_k(results, q["expect_docs"], args.k)
            hits += 1 if ok else 0
            details.append(
                {
                    "query": q["query"],
                    "hit": ok,
                    "top_doc_ids": [r.get("doc_id") for r in results[: args.k]],
                },
            )

        recall = hits / len(queries) if queries else 0.0
        print(
            json.dumps(
                {
                    "queries": len(queries),
                    "hits": hits,
                    "recall_at_k": recall,
                    "k": args.k,
                    "details": details,
                    "note": "With EMBEDDING_BACKEND=demo, recall is not semantic—use real backends for meaningful scores.",
                },
                indent=2,
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
