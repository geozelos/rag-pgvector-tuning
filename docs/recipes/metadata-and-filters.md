# Recipe: Metadata on chunks and filters on retrieve

**Goal:** Store a JSON object per chunk and restrict search with **`metadata_filter`**.

## Prerequisite

Migrations must include the `metadata` column (see `migrations/004_chunks_metadata.sql`). Run:

```bash
make migrate
```

(uses `DATABASE_URL`)

## Ingest with metadata

```bash
curl -s -X POST http://127.0.0.1:8000/ingest/chunks \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [{
      "tenant_id": "demo",
      "source_type": "doc",
      "doc_id": "faq-1",
      "chunk_index": 0,
      "content": "Returns are accepted within 30 days.",
      "metadata": {"section": "returns", "lang": "en"}
    }]
  }'
```

## Retrieve with containment filter

Postgres uses **`metadata @>` filter** (your filter object must be contained in the row’s `metadata`).

```bash
curl -s -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "return policy",
    "k": 5,
    "tenant_id": "demo",
    "metadata_filter": {"section": "returns"}
  }'
```

## CLI

```bash
uv run rag-cli retrieve --query "return policy" --k 5 --tenant-id demo \
  --metadata-filter '{"section":"returns"}'
```

## Notes

- Empty `metadata` on ingest is stored as `{}`.
- Combine with `tenant_id` / `source_type` for narrower candidate sets before vector ordering.
