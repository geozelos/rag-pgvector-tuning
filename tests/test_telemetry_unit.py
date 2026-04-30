"""TelemetryCollector behavior."""

from __future__ import annotations

from rag.profiles import Guardrails
from rag.telemetry import TelemetryCollector, assert_param_whitelisted


def test_emit_ingest_increments_backlog_on_failures() -> None:
    col = TelemetryCollector()
    col.emit_ingest(batch_size=2, duration_ms=5.0, failures=0)
    assert col.summary()["ingest_backlog_batches_estimate"] == 0.0
    col.emit_ingest(batch_size=1, duration_ms=1.0, failures=3)
    assert col.summary()["ingest_backlog_batches_estimate"] == 3.0


def test_emit_retrieve_populates_latency_percentiles() -> None:
    col = TelemetryCollector()
    for ms in [10.0, 20.0, 30.0, 40.0, 50.0]:
        col.emit_retrieve(
            duration_ms=ms,
            k=5,
            index_family="hnsw",
            ef_search=20,
            ivfflat_probes=None,
            filter_tenant_id=None,
            filter_source_type=None,
        )
    s = col.summary()
    assert s["retrieve_recent_events"] == 5
    assert s["retrieve_latency_ms_p50"] is not None
    assert s["retrieve_latency_ms_p99"] is not None


def test_format_log_line_is_json_compact() -> None:
    line = TelemetryCollector.format_log_line({"event": "x", "k": 1})
    assert '"event":"x"' in line


def test_assert_param_whitelisted_delegates() -> None:
    g = Guardrails(whitelist_runtime_params=["hnsw.ef_search"])
    assert_param_whitelisted("hnsw.ef_search", g)
