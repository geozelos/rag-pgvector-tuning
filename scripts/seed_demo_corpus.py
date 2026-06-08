#!/usr/bin/env python3
"""
Seed a reproducible demo corpus via POST /ingest/chunks for latency benchmarks.

Uses **httpx** from the default install (**`uv sync`**).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import httpx

DEFAULT_TOPICS = (
    "pgvector stores embeddings in PostgreSQL for semantic search.",
    "HNSW ef_search trades retrieval latency against recall at query time.",
    "IVFFlat probes control how many index lists are scanned per query.",
    "RAG pipelines retrieve relevant chunks before calling an LLM.",
    "Vector indexes accelerate nearest-neighbor search over embeddings.",
    "Cosine distance is common for normalized embedding vectors.",
    "Metadata filters narrow candidates before vector ordering.",
    "YAML profiles version default search knobs for different workloads.",
    "Telemetry percentiles guide automated tuning of session parameters.",
    "Postgres extensions keep retrieval close to transactional data.",
)


def _chunk_rows(*, total: int, tenant_id: str, source_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    topics = list(DEFAULT_TOPICS)
    for i in range(total):
        topic = topics[i % len(topics)]
        doc_id = f"bench-doc-{i // 5:04d}"
        chunk_index = i % 5
        rows.append(
            {
                "tenant_id": tenant_id,
                "source_type": source_type,
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "content": f"{topic} chunk {i} for benchmark corpus sizing.",
                "metadata": {"bench": True, "seq": i},
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
        description="Seed synthetic chunks for pgvector latency benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --chunks 400
  %(prog)s --base-url http://127.0.0.1:8000 --tenant-id demo --chunks 300 --json
""",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API origin")
    parser.add_argument("--chunks", type=int, default=400, help="Total chunks to upsert")
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
