"""Additional PhysicalTuner and apply_search_session_params coverage."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rag.config_loader import load_profiles_bundle
from rag.profiles import Guardrails, PhysicalProfile, ProfilesFile
from rag.telemetry import TelemetryCollector
from rag.tuner import PhysicalTuner, RuntimeOverrides, apply_search_session_params
from tests.helpers import make_ivfflat_tuner


def _hnsw_tuner() -> PhysicalTuner:
    return PhysicalTuner(
        profiles=load_profiles_bundle().profiles_doc,
        guardrails=load_profiles_bundle().guardrails,
    )


def test_recommend_ingest_backlog_gate() -> None:
    tuner = _hnsw_tuner()
    telemetry = TelemetryCollector()
    telemetry.ingest_batches_pending_estimate = 10.0
    for _ in range(20):
        telemetry.emit_retrieve(
            duration_ms=50.0,
            k=5,
            index_family="hnsw",
            ef_search=40,
            ivfflat_probes=None,
            filter_tenant_id=None,
            filter_source_type=None,
        )
    rec = tuner.recommend(telemetry)
    assert rec["rollback_signal"] is True
    assert "backlog" in rec["reason"]


def test_recommend_insufficient_without_samples() -> None:
    tuner = _hnsw_tuner()
    telemetry = TelemetryCollector()
    rec = tuner.recommend(telemetry)
    assert rec["action"] == "hold"
    # Guardrails from YAML set target_p99; with no retrieve samples p99 is missing.
    assert rec["reason"] == "need_target_p99_and_samples"


def test_recommend_no_target_latency_in_guardrails() -> None:
    tuner = PhysicalTuner(
        profiles=load_profiles_bundle().profiles_doc,
        guardrails=Guardrails(target_p99_latency_ms=None),
    )
    telemetry = TelemetryCollector()
    for _ in range(10):
        telemetry.emit_retrieve(
            duration_ms=50.0,
            k=5,
            index_family="hnsw",
            ef_search=40,
            ivfflat_probes=None,
            filter_tenant_id=None,
            filter_source_type=None,
        )
    rec = tuner.recommend(telemetry)
    assert rec["action"] == "hold"
    assert rec["reason"] == "need_target_p99_and_samples"


def test_recommend_increase_ef_when_fast_enough() -> None:
    tuner = _hnsw_tuner()
    telemetry = TelemetryCollector()
    for _ in range(50):
        telemetry.emit_retrieve(
            duration_ms=20.0,
            k=10,
            index_family="hnsw",
            ef_search=40,
            ivfflat_probes=None,
            filter_tenant_id=None,
            filter_source_type=None,
        )
    rec = tuner.recommend(telemetry)
    assert rec["index_family"] == "hnsw"
    assert rec["action"] == "increase_ef_search"
    assert rec["proposed_hnsw_ef_search"] == 44


def test_recommend_latency_regression_flag() -> None:
    tuner = _hnsw_tuner()
    telemetry = TelemetryCollector()
    for _ in range(50):
        telemetry.emit_retrieve(
            duration_ms=20.0,
            k=10,
            index_family="hnsw",
            ef_search=40,
            ivfflat_probes=None,
            filter_tenant_id=None,
            filter_source_type=None,
        )
    tuner.pre_change_latency_p99_snapshot = 10.0
    for _ in range(50):
        telemetry.emit_retrieve(
            duration_ms=200.0,
            k=10,
            index_family="hnsw",
            ef_search=40,
            ivfflat_probes=None,
            filter_tenant_id=None,
            filter_source_type=None,
        )
    rec = tuner.recommend(telemetry)
    assert rec.get("rollback_signal") is True


def test_set_override_clamp_ef_raises() -> None:
    tuner = _hnsw_tuner()
    with pytest.raises(ValueError, match="out of bounds"):
        tuner.set_override(hnsw_ef_search=5)


def test_set_override_clamp_probes_raises() -> None:
    tuner, _p, _g = make_ivfflat_tuner()
    with pytest.raises(ValueError, match="out of bounds"):
        tuner.set_override(ivfflat_probes=100)


def test_maybe_apply_without_auto_returns_rec() -> None:
    tuner = _hnsw_tuner()
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
    out = tuner.maybe_apply_from_recommendation(telemetry, auto_apply=False)
    assert "action" in out
    assert out.get("applied") is None


def test_maybe_apply_skips_on_hold() -> None:
    tuner = _hnsw_tuner()
    telemetry = TelemetryCollector()
    out = tuner.maybe_apply_from_recommendation(telemetry, auto_apply=True)
    assert out["action"] == "hold"


def test_physical_tuner_clear_overrides() -> None:
    tuner = _hnsw_tuner()
    tuner.set_override(hnsw_ef_search=48)
    assert tuner.overrides.hnsw_ef_search == 48
    tuner.clear_overrides()
    assert tuner.overrides.hnsw_ef_search is None


def test_ivfflat_recommend_increase_probes() -> None:
    tuner, _p, _g = make_ivfflat_tuner()
    telemetry = TelemetryCollector()
    for _ in range(50):
        telemetry.emit_retrieve(
            duration_ms=20.0,
            k=10,
            index_family="ivfflat",
            ef_search=None,
            ivfflat_probes=10,
            filter_tenant_id=None,
            filter_source_type=None,
        )
    rec = tuner.recommend(telemetry)
    assert rec["action"] == "increase_ivfflat_probes_for_recall"
    assert rec["proposed_ivfflat_probes"] == 12


def test_ivfflat_recommend_decrease_probes() -> None:
    tuner, _p, _g = make_ivfflat_tuner()
    telemetry = TelemetryCollector()
    for _ in range(80):
        telemetry.emit_retrieve(
            duration_ms=250.0,
            k=10,
            index_family="ivfflat",
            ef_search=None,
            ivfflat_probes=10,
            filter_tenant_id=None,
            filter_source_type=None,
        )
    rec = tuner.recommend(telemetry)
    assert rec["action"] == "decrease_ivfflat_probes_for_latency"
    assert rec["proposed_ivfflat_probes"] == 8


def test_ivfflat_step_applies_probes() -> None:
    tuner, _p, guardrails = make_ivfflat_tuner()
    data = guardrails.model_dump()
    data["cooldown_seconds_between_changes"] = 0.0
    tuner.guardrails = Guardrails.model_validate(data)
    telemetry = TelemetryCollector()
    for _ in range(80):
        telemetry.emit_retrieve(
            duration_ms=250.0,
            k=10,
            index_family="ivfflat",
            ef_search=None,
            ivfflat_probes=10,
            filter_tenant_id=None,
            filter_source_type=None,
        )
    out = tuner.maybe_apply_from_recommendation(telemetry, auto_apply=True)
    assert out.get("applied") is True
    assert tuner.overrides.ivfflat_probes == 8


@pytest.mark.asyncio
async def test_apply_search_session_params_hnsw() -> None:
    conn = AsyncMock()
    prof = PhysicalProfile(
        index_family="hnsw",
        search={"hnsw_ef_search": 33},
    )
    await apply_search_session_params(conn, prof, RuntimeOverrides())
    conn.execute.assert_awaited()
    args = conn.execute.await_args[0]
    assert "hnsw.ef_search" in args[0]


@pytest.mark.asyncio
async def test_apply_search_session_params_ivfflat() -> None:
    conn = AsyncMock()
    prof = PhysicalProfile(
        index_family="ivfflat",
        search={"ivfflat_probes": 4},
    )
    await apply_search_session_params(conn, prof, RuntimeOverrides())
    args = conn.execute.await_args[0]
    assert "ivfflat.probes" in args[0]
