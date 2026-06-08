#!/usr/bin/env python3
"""Render README demo PNGs from live retrieve responses."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "docs" / "assets"


def _headers(api_key: str | None) -> dict[str, str] | None:
    if not api_key:
        return None
    return {"X-API-Key": api_key}


def _patch_ef_search(base_url: str, ef_search: int, api_key: str | None, timeout_s: float) -> None:
    url = base_url.rstrip("/") + "/config/runtime-search"
    with httpx.Client(timeout=timeout_s) as client:
        r = client.patch(url, json={"hnsw_ef_search": ef_search}, headers=_headers(api_key))
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
    tenant_id: str,
    api_key: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/retrieve"
    body = {"query": query, "k": k, "tenant_id": tenant_id}
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=body, headers=_headers(api_key))
        r.raise_for_status()
        return r.json()


def _compact_response(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results") or []
    preview = [
        {
            "doc_id": row.get("doc_id"),
            "cosine_sim": row.get("cosine_sim"),
            "content": (row.get("content") or "")[:72] + "…",
        }
        for row in results[:2]
    ]
    return {
        "duration_ms": data.get("duration_ms"),
        "profile": data.get("profile"),
        "hnsw_ef_search": data.get("hnsw_ef_search"),
        "k": data.get("k"),
        "results_preview": preview,
    }


def _render_png(lines: list[str], out_path: Path, *, width: int = 1120, line_height: int = 28) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("Install Pillow: uv run --with pillow python scripts/generate_readme_assets.py") from exc

    font = ImageFont.load_default()
    height = 40 + len(lines) * line_height + 30
    img = Image.new("RGB", (width, height), color=(18, 18, 20))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, 34), fill=(35, 35, 40))
    draw.text((16, 10), "rag-pgvector-tuning — POST /retrieve", fill=(220, 220, 220), font=font)
    y = 48
    for line in lines:
        draw.text((20, y), line, fill=(210, 230, 210), font=font)
        y += line_height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _json_lines(label: str, payload: dict[str, Any]) -> list[str]:
    text = json.dumps(payload, indent=2)
    wrapped = textwrap.indent(text, "  ")
    return [label, *wrapped.splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate README PNG assets from retrieve responses.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--tenant-id", default="demo")
    parser.add_argument("--query", default="What is ef_search?")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    low_ef, high_ef = 24, 96
    _patch_ef_search(args.base_url, low_ef, args.api_key, args.timeout)
    low = _compact_response(
        _retrieve(
            base_url=args.base_url,
            query=args.query,
            k=args.k,
            tenant_id=args.tenant_id,
            api_key=args.api_key,
            timeout_s=args.timeout,
        )
    )
    _patch_ef_search(args.base_url, high_ef, args.api_key, args.timeout)
    high = _compact_response(
        _retrieve(
            base_url=args.base_url,
            query=args.query,
            k=args.k,
            tenant_id=args.tenant_id,
            api_key=args.api_key,
            timeout_s=args.timeout,
        )
    )

    single_lines = [
        "$ curl -s -X POST /retrieve -d '{\"query\":\"What is ef_search?\",\"k\":5,\"tenant_id\":\"demo\"}'",
        "",
        *_json_lines("Response:", low)[1:],
    ]
    compare_lines = [
        f"ef_search={low_ef}",
        json.dumps(low, indent=2),
        "",
        f"ef_search={high_ef}",
        json.dumps(high, indent=2),
    ]

    retrieve_path = ASSETS_DIR / "retrieve-duration.png"
    compare_path = ASSETS_DIR / "ef-search-comparison.png"
    _render_png(single_lines, retrieve_path)
    _render_png(compare_lines, compare_path, width=1240)
    _clear_overrides(args.base_url, args.api_key, args.timeout)
    print(f"Wrote {retrieve_path}")
    print(f"Wrote {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
