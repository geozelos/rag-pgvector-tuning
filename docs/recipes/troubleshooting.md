# Recipe: Troubleshooting

## Embedding dimension errors

**Symptom:** Ingest or retrieve fails mentioning embedding length vs configured dimension.

**Fix:** Align **`config/embedding.yaml`** `embedding_dim` with the model / server you use under **`EMBEDDING_BACKEND`**. Restart the API after changing YAML or env.

## `/ready` returns 503

**Symptom:** `chunks_table: false` or `pgvector: false`.

**Fix:** Run **`make migrate`** (or **`uv run python scripts/migrate.py`**) against the same **`DATABASE_URL`** the API uses. Ensure the Postgres image has **pgvector** (Compose file uses `pgvector/pgvector`).

## Cannot connect to Postgres from host

**Symptom:** `connection refused` on `localhost:5433`.

**Fix:** Start DB with **`make postgres-only`** or **`make up`**. Confirm **`make ps`** shows Postgres healthy.

## `docker compose` vs `docker-compose`

**Symptom:** `unknown flag` or command not found.

**Fix:** Use V2 **`docker compose`** or legacy **`docker-compose`** consistently. With **Makefile**: `make up COMPOSE=docker-compose`.

## Integration tests skipped

**Symptom:** pytest reports integration skipped.

**Fix:** Start Postgres and set **`INTEGRATION_DATABASE_URL`** if not using default `postgresql://rag:rag@localhost:5433/rag`. Then **`make integration`**.

## OpenAPI contract test failure

**Symptom:** `tests/test_openapi_contract.py` diffs against `tests/fixtures/openapi.json`.

**Fix:** If the API change is intentional:

```bash
make openapi-snapshot
```

Then commit the updated JSON.

## Tenant / upsert confusion

**Symptom:** Unexpected overwrites or cross-tenant reads.

**Fix:** Upsert key is **`(doc_id, chunk_index)`**—not unique per tenant in the default schema. Use filters and read **SECURITY.md** (multi-tenancy is client-declared). Consider **`REQUIRE_TENANT_ID=true`** for retrieve.
