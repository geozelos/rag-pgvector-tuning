# Documentation

The main **[README](../README.md)** is the entry point (quick start, configuration tables, project layout). Use this folder for **focused recipes**—copy-paste flows you can adapt.

## Recipes

| Recipe | When to use |
| ------ | ----------- |
| [Full stack in Docker](recipes/full-stack-docker.md) | API + Postgres via Compose; quickest demo |
| [Local API, Docker Postgres](recipes/local-api-docker-postgres.md) | Debug Python on the host, DB in a container |
| [Real embeddings (OpenAI / local)](recipes/real-embeddings.md) | Swap out hash `demo` vectors for semantic search |
| [Metadata storage and filters](recipes/metadata-and-filters.md) | JSON `metadata` on chunks + `metadata_filter` on retrieve |
| [IVFFlat profile and migrations](recipes/ivfflat-profile.md) | Move from default HNSW to IVFFlat-oriented setup |
| [Load testing and tuner](recipes/load-testing-and-tuner.md) | Sustained `/retrieve` traffic for `/tuner/recommend` |
| [Recall smoke script](recipes/eval-recall.md) | Tiny labeled ingest + retrieve check |
| [API hardening checklist](recipes/api-hardening.md) | Tenant requirement, CORS, auth, rate limit, hide `/docs` |
| [Troubleshooting](recipes/troubleshooting.md) | Common errors (dimension, migrations, DB URL, Compose) |

## Security and deployment edge

Production-oriented notes (reverse proxy, OWASP mapping) stay in **[SECURITY.md](../SECURITY.md)**.

## Operator shortcuts

See the root **[Makefile](../Makefile)** (`make help`) and the **Operator ergonomics** section in the main README.
