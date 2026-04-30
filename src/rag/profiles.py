"""
Pydantic models for **physical profiles** and **tuner guardrails** (loaded from YAML).

- **Profiles** describe which index family (HNSW vs IVFFlat) the app assumes and default search knobs.
- **Guardrails** bound how far the MVP tuner may move those knobs and when to signal rollback.

SQL migrations define the actual index type in PostgreSQL; profiles must stay **consistent** with DB state.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProfileBuild(BaseModel):
    """Index **build-time** hints documented in YAML (apply via migrations / DBA playbooks, not auto here)."""

    m: int | None = None
    ef_construction: int | None = None
    lists: int | None = None


class ProfileSearch(BaseModel):
    """Per-query defaults: HNSW ``ef_search`` and/or IVFFlat ``probes`` (one may be null)."""

    hnsw_ef_search: int | None = None
    ivfflat_probes: int | None = None


class PhysicalProfile(BaseModel):
    """Named tuning preset: index family, optional build notes, and search defaults."""

    description: str | None = None
    index_family: Literal["hnsw", "ivfflat"]
    build: ProfileBuild = Field(default_factory=ProfileBuild)
    search: ProfileSearch = Field(default_factory=ProfileSearch)


class ProfilesFile(BaseModel):
    """Parsed ``config/profiles.yaml`` root: picks ``active_profile`` and maps names to :class:`PhysicalProfile`."""

    schema_version: str = "1"
    active_profile: str
    profiles: dict[str, PhysicalProfile]

    def get_active_pair(self) -> tuple[str, PhysicalProfile]:
        """Return ``(profile_name, profile)`` for :attr:`active_profile` or raise ``KeyError``."""
        if self.active_profile not in self.profiles:
            raise KeyError(f"Unknown active_profile={self.active_profile!r}")
        return self.active_profile, self.profiles[self.active_profile]


class Guardrails(BaseModel):
    """Parsed ``config/tuner_guardrails.yaml``: limits and behavioral thresholds for :class:`rag.tuner.PhysicalTuner`."""

    schema_version: str = "1"
    max_delta_ef_search_per_step: int = 8
    max_delta_ivfflat_probes_per_step: int = 2
    min_ef_search: int = 16
    max_ef_search: int = 200
    min_ivfflat_probes: int = 1
    max_ivfflat_probes: int = 50
    cooldown_seconds_between_changes: float = 60.0
    rollback_p99_latency_ms_multiplier: float = 1.25
    rollback_ingest_backlog_threshold_batches: int = 3
    canary_fraction: float = 0.0
    whitelist_runtime_params: list[str] = Field(
        default_factory=lambda: ["hnsw.ef_search", "ivfflat.probes"]
    )
    target_p99_latency_ms: float | None = None


def validate_whitelist_param(name: str, guardrails: Guardrails) -> None:
    """Ensure ``name`` is allowed for runtime patches (e.g. ``hnsw.ef_search``)."""
    if name not in guardrails.whitelist_runtime_params:
        msg = f"Param {name!r} is not in tuner whitelist_runtime_params"
        raise ValueError(msg)
