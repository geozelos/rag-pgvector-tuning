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


def test_content_for_rank_similarity_decreases_with_rank() -> None:
    """Adjacent ranks stay ordered, including past the old noise-count wrap."""
    stem = _seed.CLUSTER_STEMS[0]
    tokens = stem.split()
    dim = 768
    query = demo_embedding(stem, dim)
    n_clusters = len(_seed.CLUSTER_STEMS)
    ranks = list(range(0, 30)) + [78, 79, 80, 81, 119, 120, 121, 400]
    sims: list[float] = []
    for rank in ranks:
        seq = rank * n_clusters
        text = _seed.content_for_rank(tokens, rank, seq)
        sims.append(_cosine(query, demo_embedding(text, dim)))
    for earlier, later in zip(sims, sims[1:]):
        assert earlier > later
    assert sims[0] > 0.95
    assert sims[-1] < 0.7
