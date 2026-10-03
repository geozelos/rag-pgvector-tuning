"""Seed corpus geometry: closer ranks stay nearer the cluster stem in demo embeddings."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from rag.embeddings import demo_embedding

_SPEC = importlib.util.spec_from_file_location(
    "seed_demo_corpus",
    Path(__file__).resolve().parents[1] / "scripts" / "seed_demo_corpus.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_seed = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_seed)


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def test_content_for_rank_similarity_ladder() -> None:
    stem = _seed.CLUSTER_STEMS[0]
    tokens = stem.split()
    dim = 768
    query = demo_embedding(stem, dim)
    close = _cosine(query, demo_embedding(_seed.content_for_rank(tokens, 0, 0), dim))
    mid = _cosine(query, demo_embedding(_seed.content_for_rank(tokens, 20, 100), dim))
    far = _cosine(query, demo_embedding(_seed.content_for_rank(tokens, 400, 2000), dim))
    assert close > mid > far
    assert close > 0.95
    assert far < 0.7
