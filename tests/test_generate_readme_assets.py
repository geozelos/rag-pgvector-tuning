"""README asset helpers: comparison art must be one visual line per string."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "generate_readme_assets",
    Path(__file__).resolve().parents[1] / "scripts" / "generate_readme_assets.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_assets = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_assets)


def test_comparison_lines_do_not_embed_newlines() -> None:
    low = {"hnsw_ef_search": 24, "duration_ms": 1.2, "results_preview": []}
    high = {"hnsw_ef_search": 96, "duration_ms": 2.4, "results_preview": []}
    lines = _assets._comparison_lines(24, low, 96, high)
    assert lines[0] == "ef_search=24"
    assert all("\n" not in line for line in lines)
    assert any('"hnsw_ef_search": 96' in line for line in lines)
    assert lines.index("ef_search=96") > lines.index("ef_search=24")
