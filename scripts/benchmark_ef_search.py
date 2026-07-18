#!/usr/bin/env python3
"""
Benchmark HNSW ef_search vs retrieve latency and write docs/benchmarks artifacts.

Requires a running API with seeded corpus (see scripts/seed_demo_corpus.py).
Uses API ``duration_ms`` from successful ``/retrieve`` JSON responses (not HTTP RTT).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EF_VALUES = (16, 24, 40, 64, 96)
DEFAULT_PROFILE_EF = 40


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(int(len(sorted_vals) * p / 100.0), len(sorted_vals) - 1)
    return sorted_vals[idx]


def _headers(api_key: str | None) -> dict[str, str] | None:
    if not api_key:
        return None
    return {"X-API-Key": api_key}


def _patch_ef_search(
    *,
    base_url: str,
    ef_search: int,
    api_key: str | None,
    timeout_s: float,
) -> None:
    url = base_url.rstrip("/") + "/config/runtime-search"
    with httpx.Client(timeout=timeout_s) as client:
        r = client.patch(url, json={"hnsw_ef_search": ef_search}, headers=_headers(api_key))
        r.raise_for_status()


def _clear_overrides(*, base_url: str, api_key: str | None, timeout_s: float) -> None:
    url = base_url.rstrip("/") + "/config/runtime-search"
    with httpx.Client(timeout=timeout_s) as client:
        r = client.patch(url, json={"clear_overrides": True}, headers=_headers(api_key))
        r.raise_for_status()


def _wait_ready(base_url: str, api_key: str | None, timeout_s: float, attempts: int = 30) -> None:
    url = base_url.rstrip("/") + "/ready"
    last_err: Exception | None = None
    with httpx.Client(timeout=timeout_s) as client:
        for _ in range(attempts):
            try:
                r = client.get(url, headers=_headers(api_key))
                if r.status_code == 200:
                    return
            except httpx.HTTPError as exc:
                last_err = exc
            time.sleep(1.0)
    raise RuntimeError(f"API not ready at {url}") from last_err


def _fetch_corpus_chunks(*, base_url: str, api_key: str | None, timeout_s: float) -> int:
    url = base_url.rstrip("/") + "/telemetry/summary"
    with httpx.Client(timeout=timeout_s) as client:
        r = client.get(url, headers=_headers(api_key))
        r.raise_for_status()
        data = r.json()
    return int(data.get("corpus_chunks") or 0)


async def _run_load_duration_ms(
    *,
    base_url: str,
    qps: float,
    duration_s: float,
    query: str,
    k: int,
    tenant_id: str | None,
    api_key: str | None,
    concurrency: int,
    timeout_s: float,
) -> list[tuple[int | None, float | None]]:
    """Paced POST /retrieve; record API duration_ms from JSON on 200 responses."""
    url = base_url.rstrip("/") + "/retrieve"
    body: dict[str, Any] = {"query": query, "k": k}
    if tenant_id is not None:
        body["tenant_id"] = tenant_id
    headers = _headers(api_key)

    results: list[tuple[int | None, float | None]] = []
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    async def one_call(client: httpx.AsyncClient) -> None:
        async with sem:
            try:
                r = await client.post(url, json=body, headers=headers, timeout=timeout_s)
                duration_ms: float | None = None
                if r.status_code == 200:
                    try:
                        duration_ms = float(r.json().get("duration_ms"))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        duration_ms = None
                async with lock:
                    results.append((r.status_code, duration_ms))
            except httpx.HTTPError:
                async with lock:
                    results.append((None, None))

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


def _summarize_duration_ms(rows: list[tuple[int | None, float | None]]) -> dict[str, Any]:
    ok = [ms for status, ms in rows if status == 200 and ms is not None]
    failed = [(s, ms) for s, ms in rows if s != 200 or ms is None]
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
        out["latency_ms_ok"] = {
            "min": round(sorted_ok[0], 3),
            "p50": round(_pct(sorted_ok, 50), 3),
            "p95": round(_pct(sorted_ok, 95), 3),
            "p99": round(_pct(sorted_ok, 99), 3),
            "max": round(sorted_ok[-1], 3),
            "mean": round(statistics.mean(sorted_ok), 3),
        }
    return out


def _assign_notes(results: list[dict[str, Any]], *, default_ef: int = DEFAULT_PROFILE_EF) -> None:
    with_p50 = [
        r for r in results if (r.get("latency_ms_ok") or {}).get("p50") is not None
    ]
    if not with_p50:
        for r in results:
            r["note"] = "Default profile" if r["hnsw_ef_search"] == default_ef else "—"
        return

    ranked = sorted(with_p50, key=lambda r: r["latency_ms_ok"]["p50"])
    min_p50 = ranked[0]["latency_ms_ok"]["p50"]
    max_p50 = ranked[-1]["latency_ms_ok"]["p50"]
    spread = (max_p50 - min_p50) / min_p50 if min_p50 > 0 else 0.0
    minimal_spread = spread < 0.10 and len(ranked) > 1

    for r in results:
        ef = r["hnsw_ef_search"]
        p50 = (r.get("latency_ms_ok") or {}).get("p50")
        parts: list[str] = []
        if ef == default_ef:
            parts.append("Default profile")
        if p50 is not None and p50 == min_p50 and len(ranked) > 1:
            parts.append("Lowest p50 this run")
        elif p50 is not None and p50 == max_p50 and len(ranked) > 1:
            parts.append("Highest p50 this run")
        if minimal_spread and not parts:
            parts.append("Minimal spread at this corpus size")
        r["note"] = "; ".join(parts) if parts else "—"


async def _measure(
    *,
    base_url: str,
    ef_search: int,
    qps: float,
    warmup_s: float,
    duration_s: float,
    query: str,
    k: int,
    tenant_id: str | None,
    api_key: str | None,
    concurrency: int,
    timeout_s: float,
) -> dict[str, Any]:
    _patch_ef_search(base_url=base_url, ef_search=ef_search, api_key=api_key, timeout_s=timeout_s)
    if warmup_s > 0:
        await _run_load_duration_ms(
            base_url=base_url,
            qps=qps,
            duration_s=warmup_s,
            query=query,
            k=k,
            tenant_id=tenant_id,
            api_key=api_key,
            concurrency=concurrency,
            timeout_s=timeout_s,
        )
    rows = await _run_load_duration_ms(
        base_url=base_url,
        qps=qps,
        duration_s=duration_s,
        query=query,
        k=k,
        tenant_id=tenant_id,
        api_key=api_key,
        concurrency=concurrency,
        timeout_s=timeout_s,
    )
    summary = _summarize_duration_ms(rows)
    summary["hnsw_ef_search"] = ef_search
    summary["target_qps"] = qps
    summary["warmup_s"] = warmup_s
    summary["duration_s"] = duration_s
    summary["achieved_qps"] = round(len(rows) / duration_s, 3) if duration_s else 0.0
    return summary


def _markdown_report(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    lines = [
        "# HNSW ef_search latency benchmark",
        "",
        "Reproducible retrieve latency samples for pgvector **HNSW** query-time tuning.",
        "",
        "## Setup",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| Date (UTC) | {meta['generated_at_utc']} |",
        f"| API | `{meta['base_url']}` |",
        f"| Embedding backend | `{meta['embedding_backend']}` |",
        f"| Corpus chunks | {meta['corpus_chunks']} (observed: {meta.get('corpus_chunks_observed', 'n/a')}) |",
        f"| Latency metric | {meta.get('latency_metric', 'api_duration_ms')} — p50/p99 of `/retrieve` response `duration_ms` under load |",
        f"| Index | HNSW (default migration) |",
        f"| Tenant | `{meta['tenant_id']}` |",
        f"| Query | `{meta['query']}` |",
        f"| k | {meta['k']} |",
        f"| Load | {meta['qps']} req/s for {meta['duration_s']}s per knob (warmup {meta['warmup_s']}s) |",
        "",
        "## Results",
        "",
        "| `hnsw_ef_search` | p50 (ms) | p99 (ms) | Notes |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in payload["results"]:
        lat = row.get("latency_ms_ok") or {}
        p50 = lat.get("p50", "n/a")
        p99 = lat.get("p99", "n/a")
        note = row.get("note", "")
        lines.append(f"| {row['hnsw_ef_search']} | {p50} | {p99} | {note} |")
    lines.extend(
        [
            "",
            "> **Disclaimer:** Numbers depend on hardware, corpus size, and concurrent load.",
            "> Prefer `make demo` for a latency × recall chart. This script is a load-based",
            "> latency sweep only. Demo embeddings use offline feature hashing (topical overlap).",
            "",
            "## Reproduce",
            "",
            "```bash",
            "make up",
            "make seed-demo",
            "make benchmark-latency",
            "```",
            "",
            "Or directly:",
            "",
            "```bash",
            "uv run python scripts/seed_demo_corpus.py --chunks 2000 --tenant-id demo",
            "uv run python scripts/benchmark_ef_search.py --tenant-id demo --corpus-chunks 2000",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark ef_search latency and write docs/benchmarks artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --tenant-id demo
  %(prog)s --ef-values 16,40,96 --qps 20 --duration 20 --json
  %(prog)s --dry-run --json   # preview only; does not overwrite docs/benchmarks/*
""",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--ef-values",
        default=",".join(str(v) for v in DEFAULT_EF_VALUES),
        help="Comma-separated hnsw_ef_search values",
    )
    parser.add_argument("--qps", type=float, default=15.0)
    parser.add_argument("--warmup", type=float, default=5.0, help="Warmup seconds per knob")
    parser.add_argument("--duration", type=float, default=20.0, help="Measured seconds per knob")
    parser.add_argument("--query", default="HNSW ef_search latency benchmark")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tenant-id", default="demo")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--corpus-chunks", type=int, default=400)
    parser.add_argument("--embedding-backend", default="demo")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "docs" / "benchmarks" / "results.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "docs" / "benchmarks" / "latency-ef-search.md",
    )
    parser.add_argument("--json", action="store_true", help="Print payload JSON to stdout")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview payload only; does not call API or overwrite default benchmark files",
    )
    args = parser.parse_args()

    ef_values = [int(x.strip()) for x in args.ef_values.split(",") if x.strip()]
    if not ef_values:
        parser.error("--ef-values must list at least one integer")

    concurrency = args.concurrency
    if concurrency is None:
        concurrency = max(8, min(64, int(math.ceil(args.qps))))

    if args.dry_run:
        payload: dict[str, Any] = {
            "meta": {
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "base_url": args.base_url,
                "embedding_backend": args.embedding_backend,
                "corpus_chunks": args.corpus_chunks,
                "corpus_chunks_observed": None,
                "latency_metric": "api_duration_ms",
                "tenant_id": args.tenant_id,
                "query": args.query,
                "k": args.k,
                "qps": args.qps,
                "warmup_s": args.warmup,
                "duration_s": args.duration,
                "dry_run": True,
            },
            "results": [
                {
                    "hnsw_ef_search": v,
                    "latency_ms_ok": {"p50": None, "p99": None},
                    "note": "—",
                }
                for v in ef_values
            ],
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print("Dry-run: not writing to docs/benchmarks/ (use --json to preview payload).")
        return 0

    _wait_ready(args.base_url, args.api_key, args.timeout)
    observed = _fetch_corpus_chunks(
        base_url=args.base_url, api_key=args.api_key, timeout_s=args.timeout
    )
    if observed < args.corpus_chunks:
        print(
            f"Error: corpus has {observed} chunks but --corpus-chunks expects {args.corpus_chunks}.\n"
            "Run: make seed-demo  (or uv run python scripts/seed_demo_corpus.py --chunks 400)",
            file=sys.stderr,
        )
        return 1

    results: list[dict[str, Any]] = []
    for ef in ef_values:
        summary = asyncio.run(
            _measure(
                base_url=args.base_url,
                ef_search=ef,
                qps=args.qps,
                warmup_s=args.warmup,
                duration_s=args.duration,
                query=args.query,
                k=args.k,
                tenant_id=args.tenant_id,
                api_key=args.api_key,
                concurrency=concurrency,
                timeout_s=args.timeout,
            )
        )
        if summary["requests_failed"]:
            print(
                f"Warning: ef_search={ef} had {summary['requests_failed']} failed requests",
                file=sys.stderr,
            )
        results.append(summary)
    _assign_notes(results)
    _clear_overrides(base_url=args.base_url, api_key=args.api_key, timeout_s=args.timeout)

    payload = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "base_url": args.base_url,
            "embedding_backend": args.embedding_backend,
            "corpus_chunks": args.corpus_chunks,
            "corpus_chunks_observed": observed,
            "latency_metric": "api_duration_ms",
            "tenant_id": args.tenant_id,
            "query": args.query,
            "k": args.k,
            "qps": args.qps,
            "warmup_s": args.warmup,
            "duration_s": args.duration,
            "dry_run": False,
        },
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown_report(payload), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Wrote {args.output_json}")
        print(f"Wrote {args.output_md}")

    failed = sum(r.get("requests_failed", 0) for r in payload["results"])
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
