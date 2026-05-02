"""
Offline recommendation and optional application of pgvector **query-time** parameters.

Implements a small **MVP** loop:

- **HNSW**: adjust ``hnsw.ef_search`` (higher → usually slower, better recall).
- **IVFFlat**: adjust ``ivfflat.probes`` (higher → usually slower, better recall).

Decisions use rolling latency percentiles from :class:`rag.telemetry.TelemetryCollector` and
limits from :class:`rag.profiles.Guardrails` (``config/tuner_guardrails.yaml``).

For how parameters reach PostgreSQL, see :func:`apply_search_session_params`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from rag.profiles import Guardrails, PhysicalProfile, ProfilesFile
from rag.telemetry import TelemetryCollector


@dataclass(slots=True)
class RuntimeOverrides:
    """In-memory overrides taking precedence over the active YAML profile until cleared."""

    hnsw_ef_search: int | None = None
    ivfflat_probes: int | None = None


class PhysicalTuner:
    """Load active profile + guardrails; recommend and optionally apply bounded search-knob changes.

    This is **not** a production autoscaler. It illustrates how YAML + telemetry can drive
    session ``SET LOCAL``-style parameters on each retrieve transaction.
    """

    def __init__(
        self,
        *,
        profiles: ProfilesFile,
        guardrails: Guardrails,
    ) -> None:
        self.profiles = profiles
        self.guardrails = guardrails
        self.overrides = RuntimeOverrides()
        # ``None`` means "never applied"; ``0.0`` would treat hosts with ``monotonic() < cooldown`` as still cooling down.
        self.last_change_mono: float | None = None
        self.pre_change_latency_p99_snapshot: float | None = None
        self.last_reason: str | None = None
        self.last_recommendation: dict[str, Any] | None = None

    def active(self) -> tuple[str, PhysicalProfile]:
        return self.profiles.get_active_pair()

    def effective_search(self) -> tuple[int | None, int | None]:
        _, prof = self.active()
        ef = self.overrides.hnsw_ef_search or prof.search.hnsw_ef_search
        probes = self.overrides.ivfflat_probes or prof.search.ivfflat_probes
        return ef, probes

    def set_override(
        self,
        *,
        hnsw_ef_search: int | None = None,
        ivfflat_probes: int | None = None,
    ) -> RuntimeOverrides:
        if hnsw_ef_search is not None:
            self._clamp_ef(hnsw_ef_search)
            self.overrides.hnsw_ef_search = hnsw_ef_search
        if ivfflat_probes is not None:
            self._clamp_probes(ivfflat_probes)
            self.overrides.ivfflat_probes = ivfflat_probes
        return self.overrides

    def clear_overrides(self) -> None:
        self.overrides = RuntimeOverrides()

    def _clamp_ef(self, v: int) -> None:
        g = self.guardrails
        if v < g.min_ef_search or v > g.max_ef_search:
            msg = f"hnsw_ef_search {v} out of bounds [{g.min_ef_search}, {g.max_ef_search}]"
            raise ValueError(msg)

    def _clamp_probes(self, v: int) -> None:
        g = self.guardrails
        if v < g.min_ivfflat_probes or v > g.max_ivfflat_probes:
            msg = f"ivfflat_probes {v} out of bounds [{g.min_ivfflat_probes}, {g.max_ivfflat_probes}]"
            raise ValueError(msg)

    def _cooldown_ok(self) -> bool:
        if self.last_change_mono is None:
            return True
        return (
            time.monotonic() - self.last_change_mono
            >= self.guardrails.cooldown_seconds_between_changes
        )

    def recommend(
        self,
        telemetry: TelemetryCollector,
    ) -> dict[str, Any]:
        """Return a dict with ``action`` (hold / increase / decrease), proposed values, and reasons."""
        summary = telemetry.summary()
        _, prof = self.active()
        target = self.guardrails.target_p99_latency_ms
        p99 = summary.get("retrieve_latency_ms_p99")

        rec: dict[str, Any] = {
            "action": "hold",
            "reason": "insufficient_data",
            "summary": summary,
            "index_family": prof.index_family,
            "current_effective_ef_search": self.effective_search()[0],
            "current_effective_ivfflat_probes": self.effective_search()[1],
        }

        if telemetry.ingest_batches_pending_estimate >= float(
            self.guardrails.rollback_ingest_backlog_threshold_batches,
        ):
            rec["reason"] = "ingest_backlog_high_advise_manual"
            rec["rollback_signal"] = True
            self.last_recommendation = rec
            return rec

        if p99 is None or target is None:
            rec["reason"] = "need_target_p99_and_samples"
            self.last_recommendation = rec
            return rec

        mult = float(self.guardrails.rollback_p99_latency_ms_multiplier)
        rollback_signal = False
        if (
            self.pre_change_latency_p99_snapshot is not None
            and float(p99) > float(self.pre_change_latency_p99_snapshot) * mult
        ):
            rollback_signal = True
            rec["rollback_signal"] = True
            rec["reason"] = "latency_regression_since_last_change"

        delta_ef = self.guardrails.max_delta_ef_search_per_step
        delta_pr = self.guardrails.max_delta_ivfflat_probes_per_step

        ef, probes = self.effective_search()

        action = "hold"
        proposed_ef = ef
        proposed_probes = probes

        if prof.index_family == "hnsw" and ef is not None:
            if float(p99) > target * 1.08:
                action = "decrease_ef_search"
                proposed_ef = max(
                    self.guardrails.min_ef_search,
                    int(ef) - delta_ef,
                )
            elif (
                float(p99) < target * 0.65
                and not rollback_signal
                and telemetry.summary()["retrieve_recent_events"] >= 50
            ):
                action = "increase_ef_search"
                proposed_ef = min(
                    self.guardrails.max_ef_search,
                    int(ef) + min(4, delta_ef),
                )

        if prof.index_family == "ivfflat" and probes is not None:
            if float(p99) > target * 1.08:
                action = "decrease_ivfflat_probes_for_latency"
                proposed_probes = max(
                    self.guardrails.min_ivfflat_probes,
                    int(probes) - delta_pr,
                )
            elif (
                float(p99) < target * 0.65
                and not rollback_signal
                and telemetry.summary()["retrieve_recent_events"] >= 50
            ):
                action = "increase_ivfflat_probes_for_recall"
                proposed_probes = min(
                    self.guardrails.max_ivfflat_probes,
                    int(probes) + delta_pr,
                )

        rec.update(
            action=action,
            reason="latency_vs_target",
            proposed_hnsw_ef_search=proposed_ef,
            proposed_ivfflat_probes=proposed_probes,
            rollback_signal=rollback_signal,
        )

        self.last_recommendation = rec
        return rec

    def maybe_apply_from_recommendation(
        self,
        telemetry: TelemetryCollector,
        *,
        auto_apply: bool = False,
    ) -> dict[str, Any]:
        """If ``auto_apply``, update :attr:`overrides` when recommendation is non-hold and cooldown allows."""
        rec = self.recommend(telemetry)
        if not auto_apply:
            return rec
        action = rec.get("action")
        if action == "hold" or rec.get("rollback_signal"):
            return rec
        _, prof = self.active()
        if not self._cooldown_ok():
            rec["skipped"] = "cooldown"
            return rec
        ef = rec.get("proposed_hnsw_ef_search")
        pr = rec.get("proposed_ivfflat_probes")
        summary_now = telemetry.summary()
        try:
            p99_now = summary_now.get("retrieve_latency_ms_p99")
            if p99_now is not None:
                self.pre_change_latency_p99_snapshot = float(p99_now)
        except (TypeError, ValueError):
            self.pre_change_latency_p99_snapshot = None

        cur_ef, cur_pr = self.effective_search()

        if prof.index_family == "hnsw" and isinstance(ef, int) and isinstance(cur_ef, int):
            if ef != cur_ef:
                self.set_override(hnsw_ef_search=ef)
                self.last_change_mono = time.monotonic()
                self.last_reason = f"applied_ef_search->{ef}"
                rec["applied"] = True

        if prof.index_family == "ivfflat" and isinstance(pr, int) and isinstance(cur_pr, int):
            if pr != cur_pr:
                self.set_override(ivfflat_probes=pr)
                self.last_change_mono = time.monotonic()
                self.last_reason = f"applied_ivfflat_probes->{pr}"
                rec["applied"] = True

        return rec


async def apply_search_session_params(conn: Any, profile: PhysicalProfile, overrides: RuntimeOverrides) -> None:
    """Set ``hnsw.ef_search`` or ``ivfflat.probes`` for the current DB transaction (``local=true``)."""
    ef = overrides.hnsw_ef_search or profile.search.hnsw_ef_search
    probes = overrides.ivfflat_probes or profile.search.ivfflat_probes

    if profile.index_family == "hnsw" and ef is not None:
        await conn.execute(
            "SELECT set_config('hnsw.ef_search', $1, true)",
            str(int(ef)),
        )
    if profile.index_family == "ivfflat" and probes is not None:
        await conn.execute(
            "SELECT set_config('ivfflat.probes', $1, true)",
            str(int(probes)),
        )
