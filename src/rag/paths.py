"""Repo-relative path helpers (find project root from ``src/rag/``)."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Directory containing ``pyproject.toml``, ``config/``, and ``migrations/``."""
    return Path(__file__).resolve().parents[2]
