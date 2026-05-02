#!/usr/bin/env python3
"""
Drive POST /retrieve at a steady average QPS to populate tuner telemetry or stress latency.

Uses **httpx** from the default install (**`uv sync`**). Use **`uv sync --group dev`** when you also run tests locally.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from typing import Any

import httpx


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(int(len(sorted_vals) * p / 100.0), len(sorted_vals) - 1)
    return sorted_vals[idx]


async def _run_load(
    *,
    base_url: str,
    qps: float,
    duration_s: float,
    query: str,
    k: int,
    tenant_id: str | None,
    source_type: str | None,
    api_key: str | None,
    concurrency: int,
    timeout_s: float,
) -> list[tuple[int | None, float]]:
    url = base_url.rstrip("/") + "/retrieve"
    body: dict[str, Any] = {"query": query, "k": k}
    if tenant_id is not None:
        body["tenant_id"] = tenant_id
    if source_type is not None:
        body["source_type"] = source_type
    headers: dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key

    results: list[tuple[int | None, float]] = []
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    async def one_call(client: httpx.AsyncClient) -> None:
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.post(url, json=body, headers=headers or None, timeout=timeout_s)
                dt = time.perf_counter() - t0
                async with lock:
                    results.append((r.status_code, dt))
            except httpx.HTTPError:
                dt = time.perf_counter() - t0
                async with lock:
                    results.append((None, dt))

    deadline = time.monotonic() + duration_s
    next_fire = time.monotonic()
    interval = 1.0 / qps
    tasks: list[asyncio.Task[None]] = []

    limits = httpx.Limits(max_connections=max(concurrency + 10, 32))
    async with httpx.AsyncClient(limits=limits) as client:
        while next_fire < deadline:
            await asyncio.sleep(max(0.0, next_fire - time.monotonic()))
            tasks.append(asyncio.create_task(one_call(client)))
            next_fire += interval
        await asyncio.gather(*tasks)

    return results


def _summarize(rows: list[tuple[int | None, float]]) -> dict[str, Any]:
    ok = [lat for status, lat in rows if status == 200]
    failed = [(s, lat) for s, lat in rows if s != 200]
    sorted_ok = sorted(ok)
    status_hist: dict[str, int] = {}
    for s, _ in rows:
        key = str(s) if s is not None else "error"
        status_hist[key] = status_hist.get(key, 0) + 1
    out: dict[str, Any] = {
        "requests_total": len(rows),
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "status_histogram": status_hist,
        "latency_ms_ok": {},
    }
    if sorted_ok:
        ms = [x * 1000.0 for x in sorted_ok]
        out["latency_ms_ok"] = {
            "min": round(ms[0], 3),
            "p50": round(_pct(ms, 50), 3),
            "p95": round(_pct(ms, 95), 3),
            "p99": round(_pct(ms, 99), 3),
            "max": round(ms[-1], 3),
            "mean": round(statistics.mean(ms), 3),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send paced POST /retrieve requests for load / tuner demos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --qps 20 --duration 30
  %(prog)s --base-url http://127.0.0.1:8000 --query "latency probe" --qps 5 --duration 15
  %(prog)s --api-key "$RAG_API_KEY" --tenant-id demo --qps 10 --duration 60 --json

Requires: uv sync (httpx is a runtime dependency).
""",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API origin without trailing slash (default: %(default)s)",
    )
    parser.add_argument("--qps", type=float, default=10.0, help="Target sustained requests per second")
    parser.add_argument("--duration", type=float, default=30.0, help="Run duration in seconds")
    parser.add_argument("--query", default="load test query", help="Retrieve query text")
    parser.add_argument("--k", type=int, default=10, help="Retrieve top-k")
    parser.add_argument("--tenant-id", default=None, help="Optional tenant_id filter")
    parser.add_argument("--source-type", default=None, help="Optional source_type filter")
    parser.add_argument("--api-key", default=None, help="If set, sent as X-API-Key header")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Max in-flight requests (default: max(8, min(64, ceil(qps))))",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout seconds")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON only")

    args = parser.parse_args()
    if args.qps <= 0:
        parser.error("--qps must be positive")
    if args.duration <= 0:
        parser.error("--duration must be positive")

    concurrency = args.concurrency
    if concurrency is None:
        concurrency = max(8, min(64, int(math.ceil(args.qps))))

    rows = asyncio.run(
        _run_load(
            base_url=args.base_url,
            qps=args.qps,
            duration_s=args.duration,
            query=args.query,
            k=args.k,
            tenant_id=args.tenant_id,
            source_type=args.source_type,
            api_key=args.api_key,
            concurrency=concurrency,
            timeout_s=args.timeout,
        )
    )

    summary = _summarize(rows)
    summary["target_qps"] = args.qps
    summary["duration_s"] = args.duration
    summary["achieved_qps"] = round(len(rows) / args.duration, 3)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Finished {summary['requests_total']} requests in {args.duration}s (~{summary['achieved_qps']} req/s)")
        print(f"OK: {summary['requests_ok']}  Failed/other: {summary['requests_failed']}")
        print(f"Status histogram: {summary['status_histogram']}")
        lat = summary["latency_ms_ok"]
        if lat:
            print(f"Latency OK (ms): min={lat['min']} p50={lat['p50']} p95={lat['p95']} p99={lat['p99']} max={lat['max']} mean={lat['mean']}")
        else:
            print("No successful responses — check API URL, readiness, auth, and corpus.")

    return 0 if summary["requests_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
