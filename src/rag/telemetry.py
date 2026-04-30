"""
In-process telemetry for ingest and retrieve operations.

:mod:`rag.main` emits structured events here; :class:`TelemetryCollector` keeps bounded deques
and computes simple percentiles (p50/p95/p99) used by :class:`rag.tuner.PhysicalTuner`.

This is **per-process** state (not persisted). Restarting the API clears history.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any

from rag.profiles import Guardrails
from rag.profiles import validate_whitelist_param


@dataclass(slots=True)
class IngestEvent:
    """One completed ingest batch (or failure) as seen by the API layer."""

    ts: float
    batch_size: int
    duration_ms: float
    failures: int
    tenant_id: str | None = None


@dataclass(slots=True)
class RetrieveEvent:
    """One retrieve request timing and filter metadata."""

    ts: float
    duration_ms: float
    k: int
    index_family: str
    ef_search: int | None
    ivfflat_probes: int | None
    filter_tenant_id: str | None
    filter_source_type: str | None


class TelemetryCollector:
    """Thread-safe ring buffers of recent events plus a small ingest-backlog heuristic."""

    def __init__(
        self,
        *,
        max_events: int = 2000,
    ) -> None:
        self._lock = Lock()
        self._ingest: deque[IngestEvent] = deque(maxlen=max_events)
        self._retrieve: deque[RetrieveEvent] = deque(maxlen=max_events)
        self.ingest_batches_pending_estimate: float = 0.0

    def emit_ingest(
        self,
        *,
        batch_size: int,
        duration_ms: float,
        failures: int,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        ev = IngestEvent(
            ts=time.monotonic(),
            batch_size=batch_size,
            duration_ms=duration_ms,
            failures=failures,
            tenant_id=tenant_id,
        )
        with self._lock:
            self._ingest.append(ev)
            if failures > 0:
                self.ingest_batches_pending_estimate += float(failures)
        return self.event_dict(
            "rag.ingest",
            batch_size=batch_size,
            duration_ms=duration_ms,
            failures=failures,
            tenant_id=tenant_id,
        )

    def emit_retrieve(
        self,
        *,
        duration_ms: float,
        k: int,
        index_family: str,
        ef_search: int | None,
        ivfflat_probes: int | None,
        filter_tenant_id: str | None,
        filter_source_type: str | None,
    ) -> dict[str, Any]:
        ev = RetrieveEvent(
            ts=time.monotonic(),
            duration_ms=duration_ms,
            k=k,
            index_family=index_family,
            ef_search=ef_search,
            ivfflat_probes=ivfflat_probes,
            filter_tenant_id=filter_tenant_id,
            filter_source_type=filter_source_type,
        )
        with self._lock:
            self._retrieve.append(ev)
        return self.event_dict(
            "rag.retrieve",
            duration_ms=duration_ms,
            k=k,
            index_family=index_family,
            hnsw_ef_search=ef_search,
            ivfflat_probes=ivfflat_probes,
            filter_tenant_id=filter_tenant_id,
            filter_source_type=filter_source_type,
        )

    @staticmethod
    def event_dict(event: str, **fields: Any) -> dict[str, Any]:
        payload = {"event": event, **fields}
        return payload

    @staticmethod
    def format_log_line(payload: dict[str, Any]) -> str:
        return json.dumps(payload, default=str, separators=(",", ":"))

    def summary(self, *, tail_ingest: int = 200, tail_retrieve: int = 200) -> dict[str, Any]:
        """Aggregate recent events: counts, latency percentiles for retrieve, backlog estimate."""
        with self._lock:
            ing = list(self._ingest)[-tail_ingest:]
            ret = list(self._retrieve)[-tail_retrieve:]
            pending_batches = float(self.ingest_batches_pending_estimate)
        lat = [x.duration_ms for x in ret] if ret else []

        def _pctl(xs: list[float], q: float) -> float | None:
            if not xs:
                return None
            ys = sorted(xs)
            idx = max(0, min(len(ys) - 1, int(round(q * (len(ys) - 1)))))
            return ys[idx]

        ingest_rows = sum(x.batch_size for x in ing)
        return {
            "ingest_recent_events": len(ing),
            "ingest_recent_rows": ingest_rows,
            "ingest_failures_recent": sum(x.failures for x in ing),
            "retrieve_recent_events": len(ret),
            "retrieve_latency_ms_p50": _pctl(lat, 0.50),
            "retrieve_latency_ms_p95": _pctl(lat, 0.95),
            "retrieve_latency_ms_p99": _pctl(lat, 0.99),
            "latest_retrieve_k": ret[-1].k if ret else None,
            "latest_index_family": ret[-1].index_family if ret else None,
            "ingest_backlog_batches_estimate": pending_batches,
        }


def assert_param_whitelisted(name: str, guardrails: Guardrails) -> None:
    """Raise ``ValueError`` if ``name`` is not listed in ``guardrails.whitelist_runtime_params``."""
    validate_whitelist_param(name, guardrails)
