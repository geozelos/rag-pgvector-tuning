# Recipe: Local API + Docker Postgres

**Goal:** Run **only Postgres** in Docker; run **uvicorn** on your machine (easier debugging, faster code edits).

## Steps

1. Start the database:

   ```bash
   make postgres-only
   ```

2. Point the app at the DB (matches Compose port mapping):

   ```bash
   export DATABASE_URL=postgresql://rag:rag@localhost:5433/rag
   ```

   Or copy [.env.example](../../.env.example) to `.env` and set `DATABASE_URL` there.

3. Install deps and migrate:

   ```bash
   uv sync
   uv run python scripts/migrate.py
   ```

4. Run the API:

   ```bash
   uv run uvicorn rag.main:app --host 0.0.0.0 --port 8000 --app-dir src
   ```

5. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## YAML changes

After editing `config/profiles.yaml` or `config/tuner_guardrails.yaml`, **restart uvicorn** (no image rebuild needed).

## Stop Postgres

```bash
make down
```

(or `docker compose down` / `docker-compose down` as appropriate)
