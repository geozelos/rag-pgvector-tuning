"""OpenAPI golden-file contract: compares ``app.openapi()`` to ``tests/fixtures/openapi.json``.

Regenerate the fixture after intentional API or schema changes::

    UPDATE_OPENAPI_SNAPSHOT=1 uv run pytest tests/test_openapi_contract.py -q

Drift after ``fastapi`` / ``pydantic`` upgrades is expected occasionally; commit the updated JSON.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from starlette.testclient import TestClient

OPENAPI_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "openapi.json"


def _canonical_openapi_json(schema: dict) -> str:
    return json.dumps(schema, sort_keys=True, indent=2) + "\n"


def test_openapi_schema_matches_fixture(http_client_mock_db: tuple[TestClient, object]) -> None:
    client, _ = http_client_mock_db
    schema = client.app.openapi()
    actual = _canonical_openapi_json(schema)

    if os.environ.get("UPDATE_OPENAPI_SNAPSHOT") == "1":
        OPENAPI_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        OPENAPI_FIXTURE.write_text(actual, encoding="utf-8")
        return

    expected = OPENAPI_FIXTURE.read_text(encoding="utf-8")
    assert actual == expected, (
        "OpenAPI schema drift. If the change is intentional, run:\n"
        "  UPDATE_OPENAPI_SNAPSHOT=1 uv run pytest tests/test_openapi_contract.py -q\n"
        "Then commit tests/fixtures/openapi.json."
    )
