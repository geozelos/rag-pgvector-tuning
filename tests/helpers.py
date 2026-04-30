"""Shared test utilities (importable as ``tests.helpers``)."""

from __future__ import annotations

from rag.profiles import Guardrails, PhysicalProfile, ProfilesFile
from rag.tuner import PhysicalTuner


def make_ivfflat_tuner() -> tuple[PhysicalTuner, ProfilesFile, Guardrails]:
    """Tuner on IVFFlat profile with guardrails aligned to repo defaults."""
    profiles = ProfilesFile(
        active_profile="ivf",
        profiles={
            "ivf": PhysicalProfile(
                description="test",
                index_family="ivfflat",
                search={"hnsw_ef_search": None, "ivfflat_probes": 10},
            ),
        },
    )
    guardrails = Guardrails(target_p99_latency_ms=120.0)
    tuner = PhysicalTuner(profiles=profiles, guardrails=guardrails)
    return tuner, profiles, guardrails
