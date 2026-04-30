"""
Run inside a fresh interpreter with ``DATABASE_URL`` set (used by integration tests).

Applies migrations, then exercises ingest + retrieve + health via Starlette TestClient
(runs FastAPI lifespan the same way production ASGI servers do).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from starlette.testclient import TestClient


def _run() -> None:
    root = Path(__file__).resolve().parents[1]
    dsn = os.environ["DATABASE_URL"]
    mig = subprocess.run(
        [sys.executable, str(root / "scripts" / "migrate.py"), "--dsn", dsn],
        check=False,
        capture_output=True,
        text=True,
    )
    if mig.returncode != 0:
        msg = (mig.stdout or "") + (mig.stderr or "")
        raise RuntimeError(f"migrate failed: {msg}")

    sys.path.insert(0, str(root / "src"))
    from rag.main import app

    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200

        ingest = client.post(
            "/ingest/chunks",
            json={
                "chunks": [
                    {
                        "tenant_id": "integ",
                        "source_type": "doc",
                        "doc_id": "integration-doc",
                        "chunk_index": 0,
                        "content": "Integration test chunk about pgvector.",
                    },
                ],
            },
        )
        assert ingest.status_code == 200, ingest.text

        ret = client.post(
            "/retrieve",
            json={"query": "pgvector", "k": 3, "tenant_id": "integ"},
        )
        assert ret.status_code == 200, ret.text
        body = ret.json()
        assert body["results"], "expected at least one hit"
        assert any(r.get("doc_id") == "integration-doc" for r in body["results"])

        prof = client.get("/config/active-profile")
        assert prof.status_code == 200


def main() -> None:
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL is required")
    _run()


if __name__ == "__main__":
    main()
