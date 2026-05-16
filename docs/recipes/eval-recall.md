# Recipe: Minimal recall smoke (`eval_recall.py`)

**Goal:** One command to ingest a tiny labeled set, run a couple of queries (including **`metadata_filter`**), and print whether expected **`doc_id`** values appear in the top‑**k** hits.

## Run

```bash
export RAG_BASE_URL=http://127.0.0.1:8000   # default if unset
uv run python scripts/eval_recall.py --k 5
```

Skip ingest if data already exists:

```bash
uv run python scripts/eval_recall.py --k 5 --skip-ingest
```

## Interpretation

- With **`EMBEDDING_BACKEND=demo`**, rankings are **not** semantic—**`hit: false`** is normal; the script still validates HTTP, filters, and plumbing.
- With **`openai`** or **`local`**, interpret **`recall_at_k`** in the printed JSON more seriously.

## Reference

[scripts/eval_recall.py](../../scripts/eval_recall.py)
