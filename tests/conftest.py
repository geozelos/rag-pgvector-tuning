"""Shared test helpers (do not import ``rag.main`` at collection time)."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest


class _FakeTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> bool:
        return False


class FakeConn:
    """Minimal asyncpg connection surface used by ``rag.main``."""

    def __init__(self) -> None:
        self.executemany = AsyncMock()
        self.execute = AsyncMock()
        self.fetch = AsyncMock(return_value=[])
        self.fetchval = AsyncMock(return_value=0)

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()


class _Acquire:
    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> FakeConn:
        return self._conn

    async def __aexit__(self, *args: object) -> bool:
        return False


class FakePool:
    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)

    async def close(self) -> None:
        return None


@pytest.fixture
def http_client_mock_db(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Any, FakeConn]]:
    """HTTP client with ``rag.main``'s ``asyncpg`` replaced (runs FastAPI lifespan via Starlette)."""
    import rag.main as main_mod

    conn = FakeConn()

    async def _create_pool(*_a: object, **_k: object) -> FakePool:
        return FakePool(conn)

    monkeypatch.setattr(main_mod, "asyncpg", SimpleNamespace(create_pool=_create_pool))

    from starlette.testclient import TestClient

    with TestClient(main_mod.app) as client:
        yield client, conn
