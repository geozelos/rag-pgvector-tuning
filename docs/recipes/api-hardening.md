# Recipe: API hardening checklist

**Goal:** Tighten defaults before exposing the service beyond localhost.

Use **[SECURITY.md](../../SECURITY.md)** for the full threat model. Environment variables are summarized in **[README](../../README.md)** (Operations) and **[.env.example](../../.env.example)**.

## Quick checklist

| Control | Env / behavior |
| ------- | ---------------- |
| Require tenant on retrieve | `REQUIRE_TENANT_ID=true` → missing `tenant_id` → **400** |
| Browser CORS | `CORS_ORIGINS=https://app.example.com,https://staging.example.com` |
| Hide interactive OpenAPI | `DISABLE_OPENAPI_UI=true` → `/docs` and `/redoc` off |
| Shared secret | `RAG_API_KEY` → **Bearer** or **`X-API-Key`** on routes except `/health`, `/ready` |
| Per-IP throttle (single worker) | `RATE_LIMIT_PER_MINUTE=120` → **429** (prefer proxy in production) |

## Edge termination

Put **TLS**, **global rate limits**, and **WAF** at nginx, Caddy, Traefik, or a cloud LB—see snippets in **SECURITY.md**.

## Postgres

Do not expose port **5433** publicly; rotate **`rag`/`rag`** credentials outside demos.
