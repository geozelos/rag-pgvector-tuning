"""Demo sweep helpers: applied ef_search must match the response, oracle ids are row ids."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "demo_latency_recall",
    Path(__file__).resolve().parents[1] / "scripts" / "demo_latency_recall.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_demo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_demo)


def test_recall_method_names_row_ids() -> None:
    assert _demo.RECALL_METHOD == "exact_seqscan_topk_row_id"


def test_require_applied_ef_rejects_mismatch() -> None:
    with pytest.raises(RuntimeError, match="hnsw_ef_search"):
        _demo._require_applied_ef({"hnsw_ef_search": 40}, 96)


def test_require_applied_ef_accepts_match() -> None:
    _demo._require_applied_ef({"hnsw_ef_search": 96}, 96)
