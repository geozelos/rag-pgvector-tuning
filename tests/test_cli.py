"""Tests for rag.cli (mocked HTTP)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import rag.cli as cli_mod


class _OkJsonResp:
    status_code = 200
    headers = {"content-type": "application/json"}

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _ErrResp:
    def __init__(self, status_code: int, body: dict | str) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = {"content-type": "application/json"}
        self.text = json.dumps(body) if isinstance(body, dict) else str(body)

    def json(self) -> dict:
        if isinstance(self._body, dict):
            return self._body
        raise json.JSONDecodeError("msg", "", 0)

    def raise_for_status(self) -> None:
        import httpx

        raise httpx.HTTPStatusError("fail", request=MagicMock(), response=self)


def test_cli_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        cli_mod.main(["--help"])
    assert exc.value.code == 0


def test_cli_retrieve_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def close(self) -> None:
            pass

        def post(self, path: str, json: dict | None = None, params: dict | None = None) -> _OkJsonResp:
            assert path == "/retrieve"
            assert json == {"query": "hello", "k": 3, "tenant_id": "t1"}
            return _OkJsonResp({"profile": "default", "results": []})

    monkeypatch.setattr(cli_mod.httpx, "Client", FakeClient)
    code = cli_mod.main(["retrieve", "--query", "hello", "--k", "3", "--tenant-id", "t1"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["profile"] == "default"


def test_cli_tune_step_posts_auto_apply(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def close(self) -> None:
            pass

        def post(self, path: str, json: dict | None = None, params: dict | None = None) -> _OkJsonResp:
            captured["path"] = path
            captured["params"] = params
            return _OkJsonResp({"applied": False})

    monkeypatch.setattr(cli_mod.httpx, "Client", FakeClient)
    code = cli_mod.main(["tune-step", "--auto-apply"])
    assert code == 0
    assert captured["path"] == "/tuner/step"
    assert captured["params"] == {"auto_apply": True}
    assert "applied" in json.loads(capsys.readouterr().out)


def test_cli_http_error_stderr(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class FakeClient:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def close(self) -> None:
            pass

        def post(self, path: str, json: dict | None = None, params: dict | None = None) -> _ErrResp:
            return _ErrResp(401, {"detail": "invalid or missing API key"})

    monkeypatch.setattr(cli_mod.httpx, "Client", FakeClient)
    code = cli_mod.main(["retrieve", "--query", "x"])
    assert code == 1
    err = json.loads(capsys.readouterr().err)
    assert err["status_code"] == 401


def test_cli_ingest_from_file(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "body.json"
    p.write_text(json.dumps({"chunks": [{"doc_id": "d", "chunk_index": 0, "content": "c"}]}), encoding="utf-8")

    class FakeClient:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def close(self) -> None:
            pass

        def post(self, path: str, json: dict | None = None, params: dict | None = None) -> _OkJsonResp:
            assert path == "/ingest/chunks"
            assert json["chunks"][0]["doc_id"] == "d"
            return _OkJsonResp({"upserted": 1, "duration_ms": 1.0})

    monkeypatch.setattr(cli_mod.httpx, "Client", FakeClient)
    code = cli_mod.main(["ingest", "--file", str(p)])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["upserted"] == 1
