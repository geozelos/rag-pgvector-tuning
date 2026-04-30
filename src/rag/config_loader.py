"""Load ``config/*.yaml`` into typed objects (embedding, profiles, guardrails)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rag.models_config import EmbeddingModelConfig, load_embedding_config
from rag.paths import project_root
from rag.profiles import Guardrails, ProfilesFile


@dataclass(frozen=True)
class AppYamlConfig:
    """Bundle returned to :mod:`rag.main` startup: everything needed for embedding dim + tuner."""

    embedding: EmbeddingModelConfig
    profiles_doc: ProfilesFile
    guardrails: Guardrails


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file and require a top-level mapping."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"Expected mapping in {path}"
        raise ValueError(msg)
    return data


def load_profiles_bundle(
    *,
    embedding_path: Path | None = None,
    profiles_path: Path | None = None,
    guardrails_path: Path | None = None,
) -> AppYamlConfig:
    """Load the three config files from ``config/`` under :func:`rag.paths.project_root` unless paths given."""
    root = project_root()
    ep = embedding_path or (root / "config" / "embedding.yaml")
    pp = profiles_path or (root / "config" / "profiles.yaml")
    gp = guardrails_path or (root / "config" / "tuner_guardrails.yaml")

    embedding = load_embedding_config(_read_yaml(ep))
    profiles_doc = ProfilesFile.model_validate(_read_yaml(pp))
    guardrails = Guardrails.model_validate(_read_yaml(gp))
    return AppYamlConfig(
        embedding=embedding,
        profiles_doc=profiles_doc,
        guardrails=guardrails,
    )


def load_config_defaults() -> AppYamlConfig:
    """Convenience alias for production paths (same as ``load_profiles_bundle()`` with no overrides)."""
    return load_profiles_bundle()
