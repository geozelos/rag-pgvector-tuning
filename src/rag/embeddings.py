"""
Deterministic **demo** embeddings for local development.

Replace :func:`demo_embedding` with a call to your embedding provider while keeping the same
dimension as ``config/embedding.yaml`` and the database column type.
"""

from __future__ import annotations

import hashlib

import numpy as np


def demo_embedding(content: str, dim: int) -> list[float]:
    """Return an L2-normalized float32 vector of length ``dim`` derived from SHA-256(content).

    Uses a fixed PRNG seed per content so the same string always yields the same vector
    (good for tests; not a semantic embedding model).
    """
    h = hashlib.sha256(content.encode("utf-8")).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(h[:8], "big")))
    v = rng.standard_normal(dim, dtype=np.float64)
    n = float(np.linalg.norm(v)) or 1.0
    v = v / n
    return v.astype(np.float32).tolist()
