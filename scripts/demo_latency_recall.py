#!/usr/bin/env python3
"""
Lab-wow demo: sweep HNSW ef_search and report latency + recall@k.

Zero API keys when the API uses EMBEDDING_BACKEND=demo (feature-hash embeddings).

Recall methodology: for each gold query, take top-k ``doc_id``s at ``--oracle-ef``
(default: max ef in the sweep) as ground truth, then measure fraction recovered
at each lower ``ef_search`` (mean over queries).

Writes:
  - docs/benchmarks/latency-recall-demo.md
  - docs/benchmarks/latency-recall-demo.json
  - docs/assets/latency-recall-chart.txt (ASCII)
  - docs/assets/latency-recall-chart.png (if Pillow available)
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EF_VALUES = (8, 16, 40, 96, 200)
SEED_SCRIPT = REPO_ROOT / "scripts" / "seed_demo_corpus.py"

# Queries aimed at dense clusters in scripts/seed_demo_corpus.py
GOLD_QUERIES: tuple[dict[str, str], ...] = (
    {
        "query": (
            "HNSW ef_search latency recall approximate nearest neighbor "
            "pgvector query exploration"
        ),
    },
    {
        "query": (
            "IVFFlat probes inverted lists scan approximate search "
            "postgresql vector index"
        ),
    },
    {
        "query": (
            "PostgreSQL pgvector embedding column cosine distance "
            "similarity retrieval chunks"
        ),
    },
)


def _headers(api_key: str | None) -> dict[str, str] | None:
    if not api_key:
        return None
    return {"X-API-Key": api_key}


def _wait_ready(base_url: str, api_key: str | None, timeout_s: float, attempts: int = 60) -> None:
    url = base_url.rstrip("/") + "/ready"
    with httpx.Client(timeout=timeout_s) as client:
        for _ in range(attempts):
            try:
                r = client.get(url, headers=_headers(api_key))
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
    raise RuntimeError(f"API not ready at {url}")


def _patch_ef(base_url: str, ef: int, api_key: str | None, timeout_s: float) -> None:
    url = base_url.rstrip("/") + "/config/runtime-search"
    with httpx.Client(timeout=timeout_s) as client:
        r = client.patch(url, json={"hnsw_ef_search": ef}, headers=_headers(api_key))
        r.raise_for_status()


def _clear_overrides(base_url: str, api_key: str | None, timeout_s: float) -> None:
    url = base_url.rstrip("/") + "/config/runtime-search"
    with httpx.Client(timeout=timeout_s) as client:
        r = client.patch(url, json={"clear_overrides": True}, headers=_headers(api_key))
        r.raise_for_status()


def _retrieve(
    *,
    base_url: str,
    query: str,
    k: int,
    tenant_id: str | None,
    api_key: str | None,
    timeout_s: float,
    exact: bool = False,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/retrieve"
    # Omit tenant_id on ANN sweeps: a selective tenant btree filter can make the
    # planner prefer filter+sort over HNSW, hiding ef_search effects.
    body: dict[str, Any] = {"query": query, "k": k, "exact": exact}
    if tenant_id:
        body["tenant_id"] = tenant_id
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=body, headers=_headers(api_key))
        r.raise_for_status()
        return r.json()


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(int(len(sorted_vals) * p / 100.0), len(sorted_vals) - 1)
    return sorted_vals[idx]


def _bar(value: float, *, lo: float, hi: float, width: int = 28) -> str:
    if hi <= lo or value != value:  # NaN
        return " " * width
    t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    n = int(round(t * width))
    return "█" * n + "░" * (width - n)


def _ascii_chart(rows: list[dict[str, Any]]) -> str:
    latencies = [r["p50_ms"] for r in rows if r.get("p50_ms") == r.get("p50_ms")]
    lat_lo, lat_hi = (min(latencies), max(latencies)) if latencies else (0.0, 1.0)
    if lat_hi <= lat_lo:
        lat_hi = lat_lo + 1.0
    rec_lo, rec_hi = 0.0, 1.0

    lines = [
        "ef_search → latency (p50) and recall@k vs oracle",
        "",
        f"{'ef':>6}  {'p50_ms':>8}  {'recall':>7}  latency bar / recall bar",
        "-" * 72,
    ]
    for r in rows:
        ef = r["hnsw_ef_search"]
        p50 = r["p50_ms"]
        rec = r["recall_at_k"]
        lat_bar = _bar(p50, lo=lat_lo, hi=lat_hi, width=20)
        rec_bar = _bar(rec, lo=rec_lo, hi=rec_hi, width=20)
        lines.append(
            f"{ef:>6}  {p50:8.3f}  {rec:7.3f}  L|{lat_bar}| R|{rec_bar}|"
        )
    lines.append("")
    lines.append(
        "L = p50 duration_ms (relative) · R = mean |hit∩oracle| / |oracle| over gold queries"
    )
    return "\n".join(lines) + "\n"


def _render_png(ascii_text: str, out_path: Path) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    lines = ascii_text.rstrip("\n").splitlines() or [""]
    font = ImageFont.load_default()
    line_h = 16
    width = 920
    height = 40 + len(lines) * line_h + 24
    img = Image.new("RGB", (width, height), color=(18, 18, 20))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, 28), fill=(35, 35, 40))
    draw.text(
        (12, 8),
        "rag-pgvector-tuning — ef_search latency × recall",
        fill=(220, 220, 220),
        font=font,
    )
    y = 40
    for line in lines:
        draw.text((16, y), line, fill=(200, 220, 200), font=font)
        y += line_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return True


def _row_ids(data: dict[str, Any], k: int) -> list[str]:
    """Prefer stable chunk ``id``; fall back to doc_id:chunk_index."""
    out: list[str] = []
    for row in (data.get("results") or [])[:k]:
        if row.get("id") is not None:
            out.append(str(row["id"]))
            continue
        doc_id = row.get("doc_id")
        chunk_index = row.get("chunk_index")
        if doc_id is not None:
            out.append(f"{doc_id}:{chunk_index}")
    return out


def _build_oracle(
    *,
    base_url: str,
    k: int,
    api_key: str | None,
    timeout_s: float,
) -> dict[str, list[str]]:
    """Top-k row ids per gold query via exact cosine order (``exact=true`` / seq scan)."""
    oracle: dict[str, list[str]] = {}
    for gold in GOLD_QUERIES:
        data = _retrieve(
            base_url=base_url,
            query=gold["query"],
            k=k,
            tenant_id=None,
            api_key=api_key,
            timeout_s=timeout_s,
            exact=True,
        )
        oracle[gold["query"]] = _row_ids(data, k)
    return oracle


def _recall_at_k(got: list[str], truth: list[str]) -> float:
    if not truth:
        return 0.0
    return len(set(got) & set(truth)) / float(len(truth))


def _measure_ef(
    *,
    base_url: str,
    ef: int,
    k: int,
    api_key: str | None,
    timeout_s: float,
    samples: int,
    warmup: int,
    oracle: dict[str, list[str]],
) -> dict[str, Any]:
    _patch_ef(base_url, ef, api_key, timeout_s)
    probe_query = GOLD_QUERIES[0]["query"]
    for _ in range(warmup):
        _retrieve(
            base_url=base_url,
            query=probe_query,
            k=k,
            tenant_id=None,
            api_key=api_key,
            timeout_s=timeout_s,
        )

    durations: list[float] = []
    for _ in range(samples):
        data = _retrieve(
            base_url=base_url,
            query=probe_query,
            k=k,
            tenant_id=None,
            api_key=api_key,
            timeout_s=timeout_s,
        )
        durations.append(float(data["duration_ms"]))

    recalls: list[float] = []
    for gold in GOLD_QUERIES:
        data = _retrieve(
            base_url=base_url,
            query=gold["query"],
            k=k,
            tenant_id=None,
            api_key=api_key,
            timeout_s=timeout_s,
        )
        recalls.append(_recall_at_k(_row_ids(data, k), oracle.get(gold["query"], [])))

    sorted_d = sorted(durations)
    mean_recall = statistics.mean(recalls) if recalls else float("nan")
    return {
        "hnsw_ef_search": ef,
        "p50_ms": round(_pct(sorted_d, 50), 3),
        "p99_ms": round(_pct(sorted_d, 99), 3),
        "mean_ms": round(statistics.mean(durations), 3) if durations else float("nan"),
        "samples": samples,
        "recall_at_k": round(mean_recall, 3),
        "recall_per_query": [round(x, 3) for x in recalls],
        "recall_queries": len(recalls),
    }


def _markdown(payload: dict[str, Any], chart: str) -> str:
    meta = payload["meta"]
    lines = [
        "# ef_search latency × recall demo",
        "",
        "Offline **feature-hash** embeddings + topical corpus — no API keys.",
        "",
        "**Recall:** mean fraction of exact (seq-scan) top-`k` `doc_id`s recovered "
        "at each approximate `ef_search`.",
        "",
        "## Setup",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| Date (UTC) | {meta['generated_at_utc']} |",
        f"| API | `{meta['base_url']}` |",
        f"| Tenant | `{meta['tenant_id']}` |",
        f"| k | {meta['k']} |",
        f"| Oracle | `{meta['recall_method']}` |",
        f"| Samples / ef | {meta['samples']} (+ {meta['warmup']} warmup) |",
        f"| Corpus chunks | {meta['corpus_chunks']} |",
        "",
        "## Chart",
        "",
        "```",
        chart.rstrip("\n"),
        "```",
        "",
        "## Table",
        "",
        "| `hnsw_ef_search` | p50 (ms) | p99 (ms) | recall@k |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in payload["results"]:
        lines.append(
            f"| {row['hnsw_ef_search']} | {row['p50_ms']} | {row['p99_ms']} | {row['recall_at_k']} |"
        )
    lines.extend(
        [
            "",
            "> Hardware-specific. Reproduce: `make up && make demo` "
            "(use `COMPOSE=docker-compose` if needed).",
            "",
        ]
    )
    return "\n".join(lines)


def _sync_readme_chart(chart: str) -> None:
    """Replace the fenced ASCII chart block at the top of README.md."""
    readme = REPO_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start = text.find("```text\n")
    if start < 0:
        return
    end = text.find("\n```", start)
    if end < 0:
        return
    new_block = "```text\n" + chart.rstrip("\n") + "\n```"
    updated = text[:start] + new_block + text[end + len("\n```") :]
    # Drop illustrative disclaimer if present
    updated = updated.replace(
        "> Chart from [`docs/assets/latency-recall-chart.txt`](docs/assets/latency-recall-chart.txt). "
        "Numbers are illustrative until you run `make demo` on your machine.\n\n",
        "> Measured chart (also in [`docs/assets/latency-recall-chart.txt`](docs/assets/latency-recall-chart.txt)). "
        "Reproduce: `make up && make demo`.\n\n",
    )
    readme.write_text(updated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep ef_search for latency + recall demo chart.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--tenant-id", default="demo")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--samples", type=int, default=25, help="Latency samples per ef value")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--ef-values",
        default=",".join(str(v) for v in DEFAULT_EF_VALUES),
        help="Comma-separated hnsw_ef_search values",
    )
    parser.add_argument(
        "--corpus-chunks",
        type=int,
        default=10000,
        help="Seed size when --seed is set",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed topical corpus via scripts/seed_demo_corpus.py before measuring",
    )
    parser.add_argument(
        "--sync-readme",
        action="store_true",
        help="Update the ASCII chart fence in README.md from this run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample chart without calling the API",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON payload to stdout")
    args = parser.parse_args()

    ef_values = [int(x.strip()) for x in args.ef_values.split(",") if x.strip()]
    if not ef_values:
        parser.error("--ef-values must list at least one integer")

    out_md = REPO_ROOT / "docs" / "benchmarks" / "latency-recall-demo.md"
    out_json = REPO_ROOT / "docs" / "benchmarks" / "latency-recall-demo.json"
    out_txt = REPO_ROOT / "docs" / "assets" / "latency-recall-chart.txt"
    out_png = REPO_ROOT / "docs" / "assets" / "latency-recall-chart.png"

    if args.dry_run:
        results = [
            {"hnsw_ef_search": 8, "p50_ms": 4.2, "p99_ms": 7.1, "recall_at_k": 0.400},
            {"hnsw_ef_search": 16, "p50_ms": 4.8, "p99_ms": 8.0, "recall_at_k": 0.667},
            {"hnsw_ef_search": 40, "p50_ms": 5.5, "p99_ms": 9.2, "recall_at_k": 0.867},
            {"hnsw_ef_search": 96, "p50_ms": 6.8, "p99_ms": 11.0, "recall_at_k": 1.0},
            {"hnsw_ef_search": 200, "p50_ms": 8.1, "p99_ms": 13.5, "recall_at_k": 1.0},
        ]
        chart = _ascii_chart(results)
        payload = {
            "meta": {
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "dry_run": True,
                "base_url": args.base_url,
                "tenant_id": args.tenant_id,
                "k": args.k,
                "samples": args.samples,
                "warmup": args.warmup,
                "corpus_chunks": args.corpus_chunks,
                "recall_method": "exact_seqscan_topk_doc_id",
            },
            "results": results,
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(chart)
            print("Dry-run: not writing docs/benchmarks or docs/assets.")
        return 0

    _wait_ready(args.base_url, args.api_key, args.timeout)
    if args.seed:
        cmd = [
            sys.executable,
            str(SEED_SCRIPT),
            "--base-url",
            args.base_url,
            "--chunks",
            str(args.corpus_chunks),
            "--tenant-id",
            args.tenant_id,
        ]
        if args.api_key:
            cmd.extend(["--api-key", args.api_key])
        print(f"Seeding {args.corpus_chunks} chunks …", flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    print("Building exact (seq-scan) oracle …", flush=True)
    oracle = _build_oracle(
        base_url=args.base_url,
        k=args.k,
        api_key=args.api_key,
        timeout_s=args.timeout,
    )

    results = [
        _measure_ef(
            base_url=args.base_url,
            ef=ef,
            k=args.k,
            api_key=args.api_key,
            timeout_s=args.timeout,
            samples=args.samples,
            warmup=args.warmup,
            oracle=oracle,
        )
        for ef in ef_values
    ]
    _clear_overrides(args.base_url, args.api_key, args.timeout)

    payload = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "dry_run": False,
            "base_url": args.base_url,
            "tenant_id": args.tenant_id,
            "k": args.k,
            "samples": args.samples,
            "warmup": args.warmup,
            "corpus_chunks": args.corpus_chunks,
            "recall_method": "exact_seqscan_topk_doc_id",
            "embedding_note": "demo feature-hash (offline)",
        },
        "results": results,
        "oracle_doc_ids": oracle,
    }
    chart = _ascii_chart(results)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_markdown(payload, chart), encoding="utf-8")
    out_txt.write_text(chart, encoding="utf-8")
    png_ok = _render_png(chart, out_png)
    if args.sync_readme:
        _sync_readme_chart(chart)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(chart)
        print(f"Wrote {out_md}")
        print(f"Wrote {out_json}")
        print(f"Wrote {out_txt}")
        if png_ok:
            print(f"Wrote {out_png}")
        else:
            print("Pillow not installed — skipped PNG (ASCII chart written).")
        if args.sync_readme:
            print("Updated README.md chart fence.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
