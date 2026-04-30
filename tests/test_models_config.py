"""Embedding config parsing."""

from __future__ import annotations

from rag.models_config import load_embedding_config


def test_load_embedding_config_nested_key() -> None:
    cfg = load_embedding_config({"embedding": {"model_id": "m1", "embedding_dim": 256}})
    assert cfg.model_id == "m1"
    assert cfg.embedding_dim == 256


def test_load_embedding_config_flat_mapping() -> None:
    cfg = load_embedding_config({"model_id": "m2", "embedding_dim": 128})
    assert cfg.embedding_dim == 128
