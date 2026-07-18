"""
Deterministic **demo** embeddings for local development (no API keys).

Uses signed **feature hashing** over tokens (multiple hashes per token) so texts
that share vocabulary land near each other in cosine space with enough density
for HNSW ``ef_search`` to matter in offline demos.

Replace :func:`demo_embedding` with a call to your embedding provider while keeping
the same dimension as ``config/embedding.yaml`` and the database column type.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
# Multiple feature hashes per token → denser vectors (ANN behaves more realistically).
_HASHES_PER_TOKEN = 8


def _tokens(content: str) -> list[str]:
    found = _TOKEN_RE.findall(content.lower())
    return found if found else ["empty"]


def demo_embedding(content: str, dim: int) -> list[float]:
    """Return an L2-normalized float32 vector of length ``dim`` via feature hashing.

    Each token maps to several dimensions with deterministic signs (signed hashing).
    Identical strings always yield the same vector. Topically overlapping strings
    share hashed features and therefore higher cosine similarity.
    """
    if dim <= 0:
        raise ValueError("dim must be positive")

    v = np.zeros(dim, dtype=np.float64)
    for tok in _tokens(content):
        for salt in range(_HASHES_PER_TOKEN):
            digest = hashlib.blake2b(f"{salt}:{tok}".encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            v[idx] += sign

    n = float(np.linalg.norm(v)) or 1.0
    v = v / n
    return v.astype(np.float32).tolist()
