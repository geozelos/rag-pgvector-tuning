"""Config YAML loading edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rag.config_loader import _read_yaml, load_config_defaults, load_profiles_bundle


def test_load_config_defaults_ok() -> None:
    bundle = load_config_defaults()
    assert bundle.embedding.embedding_dim >= 16
    assert bundle.profiles_doc.active_profile in bundle.profiles_doc.profiles


def test_read_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected mapping"):
        _read_yaml(p)


def test_load_profiles_bundle_custom_paths(tmp_path: Path) -> None:
    emb = tmp_path / "embedding.yaml"
    emb.write_text(
        yaml.dump({"embedding": {"model_id": "t", "embedding_dim": 128}}),
        encoding="utf-8",
    )
    prof = tmp_path / "profiles.yaml"
    prof.write_text(
        yaml.dump(
            {
                "schema_version": "1",
                "active_profile": "p",
                "profiles": {
                    "p": {
                        "index_family": "hnsw",
                        "search": {"hnsw_ef_search": 20},
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    guard = tmp_path / "guard.yaml"
    guard.write_text(yaml.dump({"schema_version": "1", "target_p99_latency_ms": 99.0}), encoding="utf-8")

    bundle = load_profiles_bundle(
        embedding_path=emb,
        profiles_path=prof,
        guardrails_path=guard,
    )
    assert bundle.embedding.embedding_dim == 128
    assert bundle.profiles_doc.active_profile == "p"
