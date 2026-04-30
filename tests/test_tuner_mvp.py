from __future__ import annotations

from rag.config_loader import load_profiles_bundle
from rag.telemetry import TelemetryCollector
from rag.tuner import PhysicalTuner


def _make_tuner() -> PhysicalTuner:
    bundle = load_profiles_bundle()
    return PhysicalTuner(profiles=bundle.profiles_doc, guardrails=bundle.guardrails)


def test_recommend_decreases_hnsw_ef_search_when_latency_above_target() -> None:
    tuner = _make_tuner()
    telemetry = TelemetryCollector()
    for _ in range(80):
        telemetry.emit_retrieve(
            duration_ms=250.0,
            k=10,
            index_family="hnsw",
            ef_search=40,
            ivfflat_probes=None,
            filter_tenant_id=None,
            filter_source_type=None,
        )
    rec = tuner.recommend(telemetry)
    assert rec["index_family"] == "hnsw"
    assert rec["action"] == "decrease_ef_search"
    assert rec["proposed_hnsw_ef_search"] == 32  # default 40 - delta 8


def test_step_respects_cooldown_when_auto_apply() -> None:
    tuner = _make_tuner()
    telemetry = TelemetryCollector()
    for _ in range(80):
        telemetry.emit_retrieve(
            duration_ms=250.0,
            k=10,
            index_family="hnsw",
            ef_search=40,
            ivfflat_probes=None,
            filter_tenant_id=None,
            filter_source_type=None,
        )
    rec1 = tuner.maybe_apply_from_recommendation(telemetry, auto_apply=True)
    assert rec1.get("applied") is True

    for _ in range(80):
        telemetry.emit_retrieve(
            duration_ms=250.0,
            k=10,
            index_family="hnsw",
            ef_search=tuner.effective_search()[0],
            ivfflat_probes=None,
            filter_tenant_id=None,
            filter_source_type=None,
        )
    rec2 = tuner.maybe_apply_from_recommendation(telemetry, auto_apply=True)
    assert rec2.get("skipped") == "cooldown"
