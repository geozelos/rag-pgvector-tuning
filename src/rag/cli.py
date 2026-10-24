"""Small HTTP CLI for ingest, retrieve, and tuner endpoints (automation-friendly)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import httpx


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2))


def _handle_response(r: httpx.Response) -> int:
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail: Any
        try:
            detail = exc.response.json()
        except json.JSONDecodeError:
            detail = exc.response.text
        print(json.dumps({"error": str(exc), "status_code": exc.response.status_code, "detail": detail}, indent=2), file=sys.stderr)
        return 1
    ct = r.headers.get("content-type", "")
    if "application/json" in ct:
        _print_json(r.json())
    else:
        print(r.text)
    return 0


def _headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"X-API-Key": api_key}


def _parse_cli_json(raw: str, *, label: str) -> Any:
    """Parse JSON from CLI args; exit with stderr message on failure."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(
            json.dumps({"error": "invalid_json", "field": label, "detail": str(exc)}),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def _client(base_url: str, api_key: str | None, timeout: float) -> httpx.Client:
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        headers=_headers(api_key),
        timeout=timeout,
    )


_SNIPPET_LEN = 80


def _passage_key(row: dict[str, Any]) -> str:
    """Stable id for one retrieve hit. Prefer the chunk row id."""
    if row.get("id") is not None:
        return str(row["id"])
    doc_id = row.get("doc_id")
    if doc_id is not None:
        return f"{doc_id}:{row.get('chunk_index')}"
    return ""


def _snippet(content: Any) -> str:
    text = str(content or "").replace("\n", " ").strip()
    if len(text) <= _SNIPPET_LEN:
        return text
    return text[:_SNIPPET_LEN] + "..."


def _format_ms(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.1f} ms"
    return "n/a"


def build_compare_report(
    *,
    query: str,
    k: int,
    ef_search: int,
    exact: dict[str, Any],
    fast: dict[str, Any],
) -> dict[str, Any]:
    """Compare one fast HNSW retrieve against the exact top-k list.

    Rank is the 1-based position in the exact result. A passage is missed when
    its id is absent from the fast result, even if some other row filled the slot.
    """
    exact_rows = list(exact.get("results") or [])[:k]
    fast_keys = {_passage_key(row) for row in list(fast.get("results") or [])[:k]}
    missed: list[dict[str, Any]] = []
    found = 0
    for rank, row in enumerate(exact_rows, start=1):
        key = _passage_key(row)
        if key and key in fast_keys:
            found += 1
            continue
        missed.append(
            {
                "rank": rank,
                "doc_id": row.get("doc_id"),
                "chunk_index": row.get("chunk_index"),
                "snippet": _snippet(row.get("content")),
            }
        )
    return {
        "query": query,
        "k": k,
        "hnsw_ef_search": ef_search,
        "fast_duration_ms": fast.get("duration_ms"),
        "full_duration_ms": exact.get("duration_ms"),
        "found": found,
        "full_count": len(exact_rows),
        "missed": missed,
    }


def format_compare_text(report: dict[str, Any]) -> str:
    """Plain-language summary of passages the fast search did not return."""
    lines = [
        f'Question: "{report["query"]}"  k={report["k"]}',
        (
            f"Fast search (ef_search={report['hnsw_ef_search']}): "
            f"{_format_ms(report['fast_duration_ms'])}   "
            f"Full search: {_format_ms(report['full_duration_ms'])}"
        ),
        (
            f"Found {report['found']} of {report['full_count']} passages "
            "the full search returns."
        ),
        "Missed:",
    ]
    missed = report["missed"]
    if not missed:
        lines.append("  (none)")
    else:
        for item in missed:
            lines.append(
                f'  #{item["rank"]}  {item["doc_id"]} / chunk {item["chunk_index"]}'
                f'   "{item["snippet"]}"'
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag-cli",
        description="Call rag-pgvector-tuning HTTP APIs from the shell.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment:
  RAG_BASE_URL   Default API origin (override with --base-url)
  RAG_API_KEY    Sent as X-API-Key when --api-key is omitted

Examples:
  rag-cli ingest --file chunks.json
  rag-cli retrieve --query "What is ef_search?" --k 5 --tenant-id demo
  rag-cli retrieve --query "What is ef_search?" --k 5 --tenant-id demo --ef-search 96
  rag-cli compare --query "What is ef_search?" --k 10 --ef-search 8 --tenant-id demo
  rag-cli tune-step --auto-apply
  curl -s chunks.json | rag-cli ingest --file -
""",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8000"),
        help="API origin (default: %(default)s or RAG_BASE_URL)",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("RAG_API_KEY") or os.environ.get("API_KEY"),
        help="X-API-Key header (default: RAG_API_KEY or API_KEY)",
    )
    p.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")

    sub = p.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="POST /ingest/chunks (JSON body)")
    ingest.add_argument(
        "--file",
        "-f",
        default="-",
        metavar="PATH",
        help='JSON path or "-" for stdin (default: %(default)s)',
    )

    retrieve = sub.add_parser("retrieve", help="POST /retrieve")
    retrieve.add_argument("--query", "-q", required=True)
    retrieve.add_argument("--k", type=int, default=10)
    retrieve.add_argument("--tenant-id", default=None)
    retrieve.add_argument("--source-type", default=None)
    retrieve.add_argument(
        "--ef-search",
        type=int,
        default=None,
        dest="hnsw_ef_search",
        metavar="N",
        help="Per-request hnsw_ef_search (does not change process-wide runtime override)",
    )
    retrieve.add_argument(
        "--metadata-filter",
        default=None,
        metavar="JSON",
        help='Optional JSON object for POST /retrieve metadata_filter (e.g. \'{"topic":"faq"}\')',
    )

    tune_step = sub.add_parser("tune-step", help="POST /tuner/step")
    tune_step.add_argument("--auto-apply", action="store_true", help="Pass auto_apply=true query param")

    compare = sub.add_parser(
        "compare",
        help="Show passages a fast HNSW search misses versus the exact top-k",
    )
    compare.add_argument("--query", "-q", required=True)
    compare.add_argument("--k", type=int, default=10)
    compare.add_argument("--tenant-id", default=None)
    compare.add_argument(
        "--ef-search",
        type=int,
        default=8,
        dest="hnsw_ef_search",
        metavar="N",
        help="Fast-search hnsw_ef_search (default: %(default)s). Does not change process state.",
    )
    compare.add_argument(
        "--json",
        action="store_true",
        help="Print the comparison as JSON instead of plain text",
    )

    return p


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv_list)

    client = _client(args.base_url, args.api_key, args.timeout)
    try:
        if args.command == "ingest":
            if args.file == "-":
                raw = sys.stdin.read()
            else:
                raw = Path(args.file).read_text(encoding="utf-8")
            body = _parse_cli_json(raw, label="ingest_body")
            r = client.post("/ingest/chunks", json=body)
            return _handle_response(r)

        if args.command == "retrieve":
            payload: dict[str, Any] = {"query": args.query, "k": args.k}
            if args.tenant_id is not None:
                payload["tenant_id"] = args.tenant_id
            if args.source_type is not None:
                payload["source_type"] = args.source_type
            if args.metadata_filter is not None:
                payload["metadata_filter"] = _parse_cli_json(
                    args.metadata_filter,
                    label="metadata_filter",
                )
            if args.hnsw_ef_search is not None:
                payload["hnsw_ef_search"] = args.hnsw_ef_search
            r = client.post("/retrieve", json=payload)
            return _handle_response(r)

        if args.command == "tune-step":
            r = client.post("/tuner/step", params={"auto_apply": args.auto_apply})
            return _handle_response(r)

        if args.command == "compare":
            payload: dict[str, Any] = {"query": args.query, "k": args.k}
            if args.tenant_id is not None:
                payload["tenant_id"] = args.tenant_id
            exact_resp = client.post("/retrieve", json={**payload, "exact": True})
            try:
                exact_resp.raise_for_status()
            except httpx.HTTPStatusError:
                return _handle_response(exact_resp)
            fast_resp = client.post(
                "/retrieve",
                json={**payload, "hnsw_ef_search": args.hnsw_ef_search},
            )
            try:
                fast_resp.raise_for_status()
            except httpx.HTTPStatusError:
                return _handle_response(fast_resp)
            report = build_compare_report(
                query=args.query,
                k=args.k,
                ef_search=args.hnsw_ef_search,
                exact=exact_resp.json(),
                fast=fast_resp.json(),
            )
            if args.json:
                _print_json(report)
            else:
                print(format_compare_text(report), end="")
            return 0

    finally:
        client.close()

    return 1


if __name__ == "__main__":
    sys.exit(main())
