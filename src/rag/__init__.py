"""
rag-pgvector-tuning

Version is taken from installed package metadata (``pyproject.toml`` / ``uv.lock``).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("rag-pgvector-tuning")
except PackageNotFoundError:
    __version__ = "0.0.0"
