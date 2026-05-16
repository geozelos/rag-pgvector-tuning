# Recipe: Real embeddings (OpenAI or local HTTP)

**Goal:** Use semantic vectors instead of the default **`demo`** hash embeddings.

## Rules

1. **`config/embedding.yaml`** — `embedding_dim` must match the model’s output size (mismatch → clear errors on ingest/retrieve).
2. Use the **same** `EMBEDDING_BACKEND` for every ingest and retrieve in a corpus you care about (mixing backends invalidates similarity).

## OpenAI

```bash
export EMBEDDING_BACKEND=openai
export OPENAI_API_KEY=sk-...
# optional: OPENAI_BASE_URL, OPENAI_EMBEDDING_MODEL
```

Set `embedding_dim` in `config/embedding.yaml` to the model dimension (e.g. `text-embedding-3-small` — use the dimension you configured with OpenAI).

Restart the API after changing env vars.

## Local OpenAI-compatible server (e.g. Ollama, TEI)

```bash
export EMBEDDING_BACKEND=local
export LOCAL_EMBEDDINGS_BASE_URL=http://127.0.0.1:11434/v1   # example; fix for your server
export LOCAL_EMBEDDING_MODEL=nomic-embed-text                 # example
```

Match `embedding_dim` to what that server returns.

## Verify

Ingest two chunks with clearly different meanings but similar length-only hash behavior would confuse `demo`; with real embeddings, retrieval order should reflect semantics more reliably.

## Reference

- [.env.example](../../.env.example)
- [src/rag/embedding_backend.py](../../src/rag/embedding_backend.py)
