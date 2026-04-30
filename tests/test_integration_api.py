"""
Integration tests against real PostgreSQL + pgvector.

Skip automatically when the DB is unreachable. Default DSN matches ``docker-compose.yml``.

Set ``INTEGRATION_DATABASE_URL`` to point at another instance.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import asyncpg
import pytest


def _integration_dsn() -> str:
    return os.environ.get(
        "INTEGRATION_DATABASE_URL",
        "postgresql://rag:rag@localhost:5433/rag",
    )


async def _postgres_reachable(dsn: str) -> bool:
    try:
        conn = await asyncpg.connect(dsn, timeout=3)
        await conn.close()
        return True
    except (OSError, asyncpg.PostgresError, TimeoutError):
        return False


@pytest.fixture(scope="module")
def postgres_integration_available() -> bool:
    return asyncio.run(_postgres_reachable(_integration_dsn()))


@pytest.mark.integration
def test_integration_ingest_and_retrieve_smoke(postgres_integration_available: bool) -> None:
    if not postgres_integration_available:
        pytest.skip("PostgreSQL not reachable (start docker compose or set INTEGRATION_DATABASE_URL)")

    env = os.environ.copy()
    env["DATABASE_URL"] = _integration_dsn()
    worker = Path(__file__).resolve().parent / "integration_worker.py"
    proc = subprocess.run(
        [sys.executable, str(worker)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
