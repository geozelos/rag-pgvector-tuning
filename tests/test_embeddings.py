"""Tests for deterministic embedding helper."""

from __future__ import annotations

import math

import numpy as np

from rag.embeddings import demo_embedding


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    return float(np.dot(va, vb))


def test_demo_embedding_shape_and_l2_unit_norm() -> None:
    dim = 128
    v = demo_embedding("same text", dim)
    assert len(v) == dim
    n = float(np.linalg.norm(np.array(v, dtype=np.float64)))
    assert math.isclose(n, 1.0, rel_tol=1e-5)


def test_demo_embedding_stable_for_same_content() -> None:
    a = demo_embedding("hello", 64)
    b = demo_embedding("hello", 64)
    assert a == b


def test_demo_embedding_differs_for_different_content() -> None:
    a = demo_embedding("aaa", 32)
    b = demo_embedding("bbb", 32)
    assert a != b


def test_demo_embedding_similar_topics_closer_than_unrelated() -> None:
    """Feature hashing must preserve topical overlap for offline recall demos."""
    dim = 256
    a = demo_embedding(
        "HNSW ef_search trades retrieval latency against recall at query time.",
        dim,
    )
    b = demo_embedding(
        "Lowering HNSW ef_search usually reduces latency and may reduce recall.",
        dim,
    )
    c = demo_embedding(
        "Tomato soup recipes need ripe tomatoes, basil, and olive oil.",
        dim,
    )
    assert _cosine(a, b) > _cosine(a, c)
