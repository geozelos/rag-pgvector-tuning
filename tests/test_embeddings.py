"""Tests for deterministic embedding helper."""

from __future__ import annotations

import math

import numpy as np

from rag.embeddings import demo_embedding


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
