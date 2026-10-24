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


def test_cli_retrieve_with_ef_search(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def close(self) -> None:
            pass

        def post(self, path: str, json: dict | None = None, params: dict | None = None) -> _OkJsonResp:
            captured["json"] = json
            return _OkJsonResp({"hnsw_ef_search": 96, "results": []})

    monkeypatch.setattr(cli_mod.httpx, "Client", FakeClient)
    code = cli_mod.main(["retrieve", "--query", "hello", "--ef-search", "96"])
    assert code == 0
    assert captured["json"] == {"query": "hello", "k": 10, "hnsw_ef_search": 96}


def test_cli_retrieve_with_metadata_filter(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def close(self) -> None:
            pass

        def post(self, path: str, json: dict | None = None, params: dict | None = None) -> _OkJsonResp:
            captured["json"] = json
            return _OkJsonResp({"results": []})

    monkeypatch.setattr(cli_mod.httpx, "Client", FakeClient)
    code = cli_mod.main(["retrieve", "--query", "hello", "--metadata-filter", '{"topic":"faq"}'])
    assert code == 0
    assert captured["json"] == {"query": "hello", "k": 10, "metadata_filter": {"topic": "faq"}}


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


def test_cli_invalid_metadata_filter_json_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeClient:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def close(self) -> None:
            pass

        def post(self, *a: object, **k: object) -> _OkJsonResp:
            calls.append("post")
            return _OkJsonResp({})

    monkeypatch.setattr(cli_mod.httpx, "Client", FakeClient)
    with pytest.raises(SystemExit) as exc:
        cli_mod.main(["retrieve", "--query", "x", "--metadata-filter", "{bad"])
    assert exc.value.code == 2
    assert calls == []


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


def test_compare_report_lists_missed_passages_by_exact_rank() -> None:
    exact = {
        "duration_ms": 14.2,
        "results": [
            {"id": 1, "doc_id": "doc-a", "chunk_index": 0, "content": "kept"},
            {"id": 2, "doc_id": "doc-a", "chunk_index": 2, "content": "first missed passage"},
            {"id": 3, "doc_id": "doc-c", "chunk_index": 0, "content": "also missed"},
        ],
    }
    fast = {
        "duration_ms": 1.1,
        "results": [
            {"id": 1, "doc_id": "doc-a", "chunk_index": 0, "content": "kept"},
            {"id": 9, "doc_id": "other", "chunk_index": 1, "content": "impostor"},
        ],
    }
    report = cli_mod.build_compare_report(
        query="where is the policy",
        k=10,
        ef_search=8,
        exact=exact,
        fast=fast,
    )
    assert report["found"] == 1
    assert report["full_count"] == 3
    assert report["fast_duration_ms"] == 1.1
    assert report["full_duration_ms"] == 14.2
    assert [m["rank"] for m in report["missed"]] == [2, 3]
    assert report["missed"][0]["doc_id"] == "doc-a"
    assert report["missed"][0]["chunk_index"] == 2
    assert report["missed"][0]["snippet"] == "first missed passage"


def test_compare_report_empty_exact_list() -> None:
    report = cli_mod.build_compare_report(
        query="nothing",
        k=10,
        ef_search=8,
        exact={"duration_ms": 2.0, "results": []},
        fast={"duration_ms": 1.0, "results": [{"id": 1, "doc_id": "x", "chunk_index": 0, "content": "nope"}]},
    )
    assert report["found"] == 0
    assert report["full_count"] == 0
    assert report["missed"] == []


def test_cli_compare_text_lists_missed(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[dict] = []

    class FakeClient:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def close(self) -> None:
            pass

        def post(self, path: str, json: dict | None = None, params: dict | None = None) -> _OkJsonResp:
            assert path == "/retrieve"
            calls.append(json or {})
            if json and json.get("exact"):
                return _OkJsonResp(
                    {
                        "duration_ms": 14.2,
                        "results": [
                            {"id": 1, "doc_id": "doc-a", "chunk_index": 0, "content": "kept"},
                            {"id": 2, "doc_id": "doc-a", "chunk_index": 2, "content": "the missed bit"},
                        ],
                    }
                )
            return _OkJsonResp(
                {
                    "duration_ms": 1.1,
                    "results": [{"id": 1, "doc_id": "doc-a", "chunk_index": 0, "content": "kept"}],
                }
            )

    monkeypatch.setattr(cli_mod.httpx, "Client", FakeClient)
    code = cli_mod.main(
        ["compare", "--query", "where is the policy", "--k", "10", "--ef-search", "8", "--tenant-id", "demo"]
    )
    assert code == 0
    assert calls[0]["exact"] is True
    assert "hnsw_ef_search" not in calls[0]
    assert calls[1]["hnsw_ef_search"] == 8
    assert calls[1].get("exact") is not True
    assert calls[0]["tenant_id"] == "demo"
    out = capsys.readouterr().out
    assert 'Question: "where is the policy"  k=10' in out
    assert "Fast search (ef_search=8): 1.1 ms" in out
    assert "Full search: 14.2 ms" in out
    assert "Found 1 of 2 passages the full search returns." in out
    assert '#2  doc-a / chunk 2   "the missed bit"' in out


def test_cli_compare_http_error_stops(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[dict] = []

    class FakeClient:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def close(self) -> None:
            pass

        def post(self, path: str, json: dict | None = None, params: dict | None = None) -> _ErrResp:
            calls.append(json or {})
            return _ErrResp(500, {"detail": "db down"})

    monkeypatch.setattr(cli_mod.httpx, "Client", FakeClient)
    code = cli_mod.main(["compare", "--query", "q", "--ef-search", "8"])
    assert code == 1
    assert len(calls) == 1
    assert calls[0]["exact"] is True
    err = json.loads(capsys.readouterr().err)
    assert err["status_code"] == 500


def test_cli_compare_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class FakeClient:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def close(self) -> None:
            pass

        def post(self, path: str, json: dict | None = None, params: dict | None = None) -> _OkJsonResp:
            if json and json.get("exact"):
                return _OkJsonResp({"duration_ms": 3, "results": []})
            return _OkJsonResp({"duration_ms": 1, "results": []})

    monkeypatch.setattr(cli_mod.httpx, "Client", FakeClient)
    code = cli_mod.main(["compare", "--query", "q", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "q"
    assert payload["found"] == 0
    assert payload["full_count"] == 0
    assert payload["missed"] == []
    assert payload["hnsw_ef_search"] == 8
