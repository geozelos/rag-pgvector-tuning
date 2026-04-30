"""Pydantic models for ``config/embedding.yaml`` (dimension and model label for the demo embedder)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmbeddingModelConfig(BaseModel):
    """Embedding settings: ``embedding_dim`` must match the ``vector(N)`` column / migration."""

    model_id: str = "demo-hash-embedding"
    embedding_dim: int = Field(ge=16, le=8192)


class EmbeddingConfigFile(BaseModel):
    """Wrapper matching YAML shape ``embedding: { ... }``."""

    embedding: EmbeddingModelConfig


def load_embedding_config(data: dict) -> EmbeddingModelConfig:
    """Extract :class:`EmbeddingModelConfig` from a mapping (either full file or inner ``embedding`` key)."""
    parsed = EmbeddingConfigFile.model_validate({"embedding": data.get("embedding", data)})
    return parsed.embedding
