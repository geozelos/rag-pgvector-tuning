"""
Apply SQL migrations in lexicographic order from ``migrations/*.sql`` (e.g. ``001_*.sql``).

- Tracks applied files in table ``schema_migrations`` (idempotent re-runs skip completed files).
- Files under ``migrations/alternate/`` are **not** run unless you pass ``--alternate-ivfflat``
  (see README: IVFFlat profile vs default HNSW migrations).

Usage::

    export DATABASE_URL=postgresql://user:pass@host:port/db
    uv run python scripts/migrate.py
    uv run python scripts/migrate.py --alternate-ivfflat
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import asyncpg


def split_sql(sql: str) -> list[str]:
    """Split on semicolon; migrations omit semicolons inside string literals."""
    parts: list[str] = []
    for chunk in sql.split(";"):
        lines = []
        for ln in chunk.splitlines():
            stripped = ln.strip()
            if not stripped or stripped.startswith("--"):
                continue
            lines.append(ln)
        stmt = "\n".join(lines).strip()
        if stmt:
            parts.append(stmt)
    return parts


async def migrate(dsn: str, migrations_dir: Path, alternate_ivfflat: bool) -> None:
    files = sorted(p for p in migrations_dir.glob("*.sql"))

    alternate = migrations_dir / "alternate" / "ivfflat_profile.sql"
    if alternate_ivfflat:
        files.append(alternate)

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              id serial PRIMARY KEY,
              filename text NOT NULL UNIQUE,
              applied_at timestamptz NOT NULL DEFAULT now()
            );
            """
        )
        for path in sorted(set(files), key=lambda p: p.name):
            fname = path.name
            applied = await conn.fetchval(
                "SELECT 1 FROM schema_migrations WHERE filename = $1", fname
            )
            if applied:
                continue
            raw = path.read_text(encoding="utf-8")
            for stmt in split_sql(raw):
                await conn.execute(stmt)
            await conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES ($1)",
                fname,
            )
            print(f"Applied {fname}")
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dsn",
        default=os.environ.get("DATABASE_URL"),
        help="postgresql DSN",
    )
    ap.add_argument(
        "--alternate-ivfflat",
        action="store_true",
        help="also apply migrations/alternate/ivfflat_profile.sql",
    )
    args = ap.parse_args()
    if not args.dsn:
        raise SystemExit("Set DATABASE_URL or pass --dsn")
    migrations_dir = Path(__file__).resolve().parent.parent / "migrations"
    asyncio.run(migrate(args.dsn, migrations_dir, args.alternate_ivfflat))


if __name__ == "__main__":
    main()
