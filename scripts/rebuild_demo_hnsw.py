#!/usr/bin/env python3
"""
Rebuild HNSW with deliberately modest build params so ``ef_search`` affects recall.

Uses DATABASE_URL (default: local compose Postgres on :5433).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import asyncpg

DEFAULT_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")


async def rebuild(*, database_url: str, m: int, ef_construction: int) -> None:
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw")
        await conn.execute("DROP INDEX IF EXISTS chunks_embedding_ivfflat")
        await conn.execute(
            f"""
            CREATE INDEX chunks_embedding_hnsw ON chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = {int(m)}, ef_construction = {int(ef_construction)})
            """
        )
    finally:
        await conn.close()


def main() -> int:
    p = argparse.ArgumentParser(description="Rebuild demo HNSW index (modest build params).")
    p.add_argument("--database-url", default=DEFAULT_URL)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--ef-construction", type=int, default=32)
    args = p.parse_args()
    if args.m < 2 or args.ef_construction < 4:
        p.error("m/ef_construction too small")
    asyncio.run(rebuild(database_url=args.database_url, m=args.m, ef_construction=args.ef_construction))
    print(f"Rebuilt HNSW (m={args.m}, ef_construction={args.ef_construction})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
