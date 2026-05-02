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


def _client(base_url: str, api_key: str | None, timeout: float) -> httpx.Client:
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        headers=_headers(api_key),
        timeout=timeout,
    )


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

    tune_step = sub.add_parser("tune-step", help="POST /tuner/step")
    tune_step.add_argument("--auto-apply", action="store_true", help="Pass auto_apply=true query param")

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
            body = json.loads(raw)
            r = client.post("/ingest/chunks", json=body)
            return _handle_response(r)

        if args.command == "retrieve":
            payload: dict[str, Any] = {"query": args.query, "k": args.k}
            if args.tenant_id is not None:
                payload["tenant_id"] = args.tenant_id
            if args.source_type is not None:
                payload["source_type"] = args.source_type
            r = client.post("/retrieve", json=payload)
            return _handle_response(r)

        if args.command == "tune-step":
            r = client.post("/tuner/step", params={"auto_apply": args.auto_apply})
            return _handle_response(r)

    finally:
        client.close()

    return 1


if __name__ == "__main__":
    sys.exit(main())
