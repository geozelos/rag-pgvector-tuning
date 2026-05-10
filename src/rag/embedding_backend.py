"""
Pluggable embedding backends: demo (hash), OpenAI HTTP API, and local OpenAI-compatible HTTP.

Configure with ``EMBEDDING_BACKEND`` and related environment variables (see :class:`rag.settings.Settings`).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

from rag.embeddings import demo_embedding
from rag.settings import Settings


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Produces dense vectors for ingest and retrieve; dimension must match YAML/database."""

    async def embed(self, text: str, dim: int) -> list[float]:
        """Return a vector of length ``dim`` suitable for the configured pgvector column."""

    async def aclose(self) -> None:
        """Release HTTP clients or other resources."""


class DemoEmbeddingBackend:
    """Deterministic hash-based vectors (see :func:`rag.embeddings.demo_embedding`)."""

    async def embed(self, text: str, dim: int) -> list[float]:
        return demo_embedding(text, dim)

    async def aclose(self) -> None:
        return None


class OpenAICompatibleHttpBackend:
    """``POST {base}/embeddings`` in OpenAI format; used for ``openai`` and ``local`` backends."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_s: float,
        require_api_key: bool,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._require_api_key = require_api_key
        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s)
        return self._client

    async def embed(self, text: str, dim: int) -> list[float]:
        if self._require_api_key and not self._api_key:
            msg = "API key is required for this embedding backend but was not configured."
            raise RuntimeError(msg)
        url = f"{self._base}/embeddings"
        client = await self._get_client()
        r = await client.post(
            url,
            headers=self._headers(),
            json={"model": self._model, "input": text},
        )
        r.raise_for_status()
        data = r.json()
        try:
            vec = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            msg = "Unexpected embeddings response shape from provider"
            raise RuntimeError(msg) from exc
        if len(vec) != dim:
            msg = (
                f"Embedding length {len(vec)} does not match configured dimension {dim} "
                f"(check config/embedding.yaml and model {self._model!r})"
            )
            raise ValueError(msg)
        return [float(x) for x in vec]

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def build_embedding_backend(settings: Settings) -> EmbeddingBackend:
    """Construct the active backend from :class:`rag.settings.Settings` (fail-fast for misconfig)."""
    name = (settings.embedding_backend or "demo").strip().lower()
    if name == "demo":
        return DemoEmbeddingBackend()
    if name == "openai":
        return OpenAICompatibleHttpBackend(
            base_url=settings.openai_base_url,
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            timeout_s=settings.embedding_http_timeout_s,
            require_api_key=True,
        )
    if name == "local":
        if not (settings.local_embeddings_base_url or "").strip():
            msg = "LOCAL_EMBEDDINGS_BASE_URL is required when EMBEDDING_BACKEND=local"
            raise RuntimeError(msg)
        return OpenAICompatibleHttpBackend(
            base_url=settings.local_embeddings_base_url,
            model=settings.local_embedding_model,
            api_key=settings.local_embeddings_api_key,
            timeout_s=settings.embedding_http_timeout_s,
            require_api_key=False,
        )
    msg = f"Unknown EMBEDDING_BACKEND: {name!r} (use demo, openai, or local)"
    raise ValueError(msg)
