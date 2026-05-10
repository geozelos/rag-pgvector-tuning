"""Embedding backend factory and demo backend."""

from __future__ import annotations

import pytest

from rag.embedding_backend import (
    DemoEmbeddingBackend,
    OpenAICompatibleHttpBackend,
    build_embedding_backend,
)
from rag.settings import Settings


@pytest.mark.asyncio
async def test_demo_backend_embed_dimension() -> None:
    b = DemoEmbeddingBackend()
    v = await b.embed("hello", 64)
    assert len(v) == 64
    await b.aclose()


def test_build_factory_demo() -> None:
    s = Settings(embedding_backend="demo")
    b = build_embedding_backend(s)
    assert isinstance(b, DemoEmbeddingBackend)


def test_build_factory_openai_returns_http_backend() -> None:
    s = Settings(
        embedding_backend="openai",
        openai_api_key="sk-test",
        openai_base_url="https://api.openai.com/v1",
    )
    b = build_embedding_backend(s)
    assert isinstance(b, OpenAICompatibleHttpBackend)
