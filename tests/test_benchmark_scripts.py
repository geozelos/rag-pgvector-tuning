"""Smoke tests for benchmark helper scripts (no running API required)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark_ef_search.py"
RESULTS_JSON = REPO_ROOT / "docs" / "benchmarks" / "results.json"


def test_benchmark_dry_run_does_not_overwrite_committed_results() -> None:
    before = RESULTS_JSON.read_text(encoding="utf-8") if RESULTS_JSON.exists() else None
    proc = subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Dry-run" in proc.stdout
    if before is not None:
        assert RESULTS_JSON.read_text(encoding="utf-8") == before


def test_benchmark_dry_run_json_preview() -> None:
    proc = subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), "--dry-run", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["meta"]["dry_run"] is True
    assert payload["meta"]["latency_metric"] == "api_duration_ms"
