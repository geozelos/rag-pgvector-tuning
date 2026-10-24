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


def test_sync_readme_chart_replaces_only_marker_block(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "intro\n\n```text\nleave me\n```\n\n"
        "<!-- latency-recall-chart:start -->\n```text\nold chart\n```\n"
        "<!-- latency-recall-chart:end -->\n\nafter\n",
        encoding="utf-8",
    )
    _demo._sync_readme_chart("new chart\n", readme)
    text = readme.read_text(encoding="utf-8")
    assert "```text\nleave me\n```" in text
    assert "old chart" not in text
    assert "```text\nnew chart\n```" in text
    assert text.endswith("after\n")


def test_ascii_for_png_replaces_glyphs_the_bitmap_font_drops() -> None:
    raw = "ef_search → latency × recall |hit∩oracle| · █░"
    safe = _demo._ascii_for_png(raw)
    assert "→" not in safe
    assert "×" not in safe
    assert "∩" not in safe
    assert "█" not in safe
    assert "->" in safe
    assert "&" in safe
    assert "#" in safe


def test_sync_readme_chart_requires_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("```text\nonly example\n```\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="latency-recall-chart"):
        _demo._sync_readme_chart("new chart", readme)
    assert "only example" in readme.read_text(encoding="utf-8")
