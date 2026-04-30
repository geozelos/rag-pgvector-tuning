"""Profile and guardrail models."""

from __future__ import annotations

import pytest

from rag.profiles import Guardrails, PhysicalProfile, ProfilesFile, validate_whitelist_param


def test_get_active_pair_unknown_raises() -> None:
    doc = ProfilesFile(
        active_profile="missing",
        profiles={
            "other": PhysicalProfile(
                index_family="hnsw",
                search={"hnsw_ef_search": 10},
            ),
        },
    )
    with pytest.raises(KeyError, match="Unknown active_profile"):
        doc.get_active_pair()


def test_validate_whitelist_param_rejects_unknown() -> None:
    g = Guardrails(whitelist_runtime_params=["hnsw.ef_search"])
    with pytest.raises(ValueError, match="not in tuner whitelist"):
        validate_whitelist_param("ivfflat.probes", g)


def test_validate_whitelist_param_accepts_listed() -> None:
    g = Guardrails(whitelist_runtime_params=["hnsw.ef_search"])
    validate_whitelist_param("hnsw.ef_search", g)
