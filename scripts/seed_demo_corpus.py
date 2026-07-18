#!/usr/bin/env python3
"""
Seed a reproducible demo corpus via POST /ingest/chunks for latency / recall demos.

Uses **httpx** from the default install (**`uv sync`**).

Builds dense topical clusters (many near-duplicate distractors) so approximate HNSW
search can diverge from exact cosine order — needed for ``ef_search`` ↔ recall demos.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import httpx

# Shared vocabulary per cluster → many near neighbors in feature-hash space.
CLUSTER_STEMS = (
    "HNSW ef_search latency recall approximate nearest neighbor pgvector query exploration",
    "IVFFlat probes inverted lists scan approximate search postgresql vector index",
    "PostgreSQL pgvector embedding column cosine distance similarity retrieval chunks",
    "metadata filter jsonb containment tenant scope before vector ordering candidates",
    "YAML profile hnsw ivfflat search knobs workload defaults guardrails tuner",
)


def _chunk_rows(*, total: int, tenant_id: str, source_type: str) -> list[dict[str, Any]]:
    """Generate ``total`` near-duplicate cluster variants (dense ANN neighborhood)."""
    rows: list[dict[str, Any]] = []
    stems = list(CLUSTER_STEMS)
    for i in range(total):
        stem = stems[i % len(stems)]
        # Slight wording drift keeps vectors close but not identical.
        content = (
            f"{stem}. Variant {i} near-duplicate cluster member for ANN stress. "
            f"Keywords echo: {stem.split()[i % 5]} {stem.split()[(i + 2) % 7]}."
        )
        doc_id = f"cluster-{i % len(stems):02d}-doc-{i // 5:05d}"
        rows.append(
            {
                "tenant_id": tenant_id,
                "source_type": source_type,
                "doc_id": doc_id,
                "chunk_index": i % 5,
                "content": content,
                "metadata": {"bench": True, "seq": i, "cluster": i % len(stems)},
            }
        )
    return rows


def _post_batches(
    *,
    base_url: str,
    chunks: list[dict[str, Any]],
    batch_size: int,
    api_key: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/ingest/chunks"
    headers: dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key

    upserted = 0
    duration_ms_total = 0.0
    batches = 0
    with httpx.Client(timeout=timeout_s) as client:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            t0 = time.perf_counter()
            r = client.post(url, json={"chunks": batch}, headers=headers or None)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            r.raise_for_status()
            data = r.json()
            upserted += int(data.get("upserted", len(batch)))
            duration_ms_total += float(data.get("duration_ms", elapsed_ms))
            batches += 1
    return {
        "chunks_requested": len(chunks),
        "upserted": upserted,
        "batches": batches,
        "ingest_duration_ms_total": round(duration_ms_total, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed dense synthetic chunks for pgvector latency / recall demos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --chunks 10000
  %(prog)s --base-url http://127.0.0.1:8000 --tenant-id demo --chunks 300 --json
""",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API origin")
    parser.add_argument("--chunks", type=int, default=10000, help="Total chunks to upsert")
    parser.add_argument("--batch-size", type=int, default=100, help="Chunks per ingest request")
    parser.add_argument("--tenant-id", default="demo", help="tenant_id for all chunks")
    parser.add_argument("--source-type", default="doc", help="source_type for all chunks")
    parser.add_argument("--api-key", default=None, help="Optional X-API-Key header")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout seconds")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON only")
    args = parser.parse_args()

    if args.chunks <= 0:
        parser.error("--chunks must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    rows = _chunk_rows(total=args.chunks, tenant_id=args.tenant_id, source_type=args.source_type)
    summary = _post_batches(
        base_url=args.base_url,
        chunks=rows,
        batch_size=args.batch_size,
        api_key=args.api_key,
        timeout_s=args.timeout,
    )
    summary["tenant_id"] = args.tenant_id
    summary["source_type"] = args.source_type

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Seeded {summary['upserted']} chunks in {summary['batches']} batches "
            f"(tenant_id={args.tenant_id!r})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
