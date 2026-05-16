# Recipe: Full stack in Docker

**Goal:** Run PostgreSQL (pgvector) and the API in containers with one command.

## Steps

1. From the repository root:

   ```bash
   make up
   ```

   If your machine only has the legacy Compose binary:

   ```bash
   make up COMPOSE=docker-compose
   ```

2. Wait until the API is healthy:

   ```bash
   make ps
   curl -s http://127.0.0.1:8000/ready
   ```

   Expect `"status":"ready"` and `chunks_table: true`.

3. Open interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

4. Optional smoke ingest + retrieve (same tenant in both calls):

   ```bash
   curl -s -X POST http://127.0.0.1:8000/ingest/chunks \
     -H "Content-Type: application/json" \
     -d '{"chunks":[{"tenant_id":"demo","source_type":"doc","doc_id":"doc-1","chunk_index":0,"content":"pgvector stores vectors in PostgreSQL."}]}'

   curl -s -X POST http://127.0.0.1:8000/retrieve \
     -H "Content-Type: application/json" \
     -d '{"query":"vectors","k":3,"tenant_id":"demo"}'
   ```

## Cleanup

```bash
make down
```

## Reference

- [docker-compose.yml](../../docker-compose.yml) — Postgres on host port **5433**, API on **8000**.
