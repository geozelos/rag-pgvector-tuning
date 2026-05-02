# Security

This repository is a **learning and demonstration** project. Treat the findings below as a **threat model and hardening checklist**, not as certification.

If you discover a security issue in this codebase, please report it **privately** to the maintainer when sensitive, or open a public issue for non-sensitive items (documentation-only, etc.).

---

## OWASP baseline

Mappings are **qualitative** (code inspection). Earlier revisions used **OWASP Top 10 (2021)** numbering; this document is maintained against **[OWASP Top 10:2025](https://owasp.org/Top10/2025/en/)**. Dependency advisories change continuously—run **`pip-audit`** (see CI and commands below), **Dependabot**, or **`uv.lock`** + OSV regularly.

---

## OWASP Top 10:2025 — review of this codebase

### A01:2025 — Broken Access Control — **high (if exposed beyond localhost)**

**Findings**

- **`tenant_id` / `source_type` are optional filters** on `POST /retrieve` — omitting them returns hits across **all** rows (`src/rag/main.py`).
- Any authenticated caller (when **`RAG_API_KEY`** / **`API_KEY`** is set) shares **one global secret** — **no RBAC**, **no per-tenant API keys**, **no JWT scopes**. With **no API key configured**, every route except **`GET /health`** and **`GET /ready`** remains **anonymous full access**.
- **`GET /health`** and **`GET /ready`** are **always unauthenticated** (for probes); **`/ready`** does not grant data access but confirms DB/pgvector/table readiness.
- **Ingest** can **overwrite** chunks by `(doc_id, chunk_index)` for **any** declared tenant string — multi-tenancy is **client-declared**, not cryptographically enforced.
- **`PATCH /config/runtime-search`**, **`POST /tuner/*`**, **`POST /telemetry/ingest-backlog/clear`** affect **global** tuning/telemetry state for the process.
- **OpenAPI `/docs` and `/redoc`** expose the attack surface.

**Mitigations (production-oriented)**

- Put the API behind an **API gateway / reverse proxy** with **mTLS or OAuth2/JWT**, **per-route scopes**, and **network policies**.
- Enforce **tenant isolation** (required tenant context + server-side checks, or PostgreSQL row-level security).
- **Disable or protect `/docs`** on untrusted networks.

---

### A02:2025 — Security Misconfiguration — **high (default stack)**

**Findings**

- **Postgres** published on host port **5433** in `docker-compose.yml` — avoid unintended exposure beyond localhost.
- **Default DB credentials** (`rag` / `rag`) are documented for dev — **not** for the internet.
- **No explicit security headers, rate limiting, or CORS allowlist** in application code (FastAPI defaults only).
- **Positive control:** **`GET /ready`** checks connectivity, **`vector`** extension, and **`public.chunks`** — helps orchestrators avoid routing traffic before migrations (does **not** replace auth).

**Mitigations**

- Bind services to **internal networks** in real deployments; **do not** expose Postgres publicly.
- Add **rate limiting** (proxy or middleware), **security headers**, and explicit **CORS** when browser clients exist.
- Use **secrets management** and **strong** credentials outside demos.

---

### A03:2025 — Software Supply Chain Failures — **ongoing**

**Findings**

- Dependencies are pinned in **`uv.lock`**; disclosed CVEs can appear at any time.
- **Docker images** should use **pinned bases** and be scanned (**Trivy**, **Grype**, etc.) before production deploy—recommended **outside** this minimal CI.

**Mitigations**

- CI runs **`pip-audit`** on exported locked runtime dependencies and **Bandit** on `src/rag` and `scripts` (see `.github/workflows/ci.yml`).
- Enable **Dependabot** or **Renovate**; **rebuild** images after upgrades.

Example (matches CI; omit **`process substitution`** for portability):

```bash
uv sync --group dev
uv export --frozen --no-dev --no-emit-project --no-hashes -o /tmp/deps-audit.txt
uv run pip-audit -r /tmp/deps-audit.txt
```

---

### A04:2025 — Cryptographic Failures — **medium (deployment-dependent)**

**Findings**

- **`DATABASE_URL`** may contain **plain-text credentials** (`.env`, Compose env).
- **No TLS** in the app—terminate **HTTPS** at the edge if needed.
- **No encryption-at-rest** configured here (use disk encryption / managed DB).

**Mitigations**

- Store secrets outside git; **rotate** passwords.
- **`sslmode=require`/`verify-full`** for Postgres clients where supported.

---

### A05:2025 — Injection — **lower for SQL; availability still matters**

**Findings**

- **SQL:** Ingest uses **parameterized** `executemany`; retrieve uses a **fixed** SQL shell with **bound parameters** for filters. Embedding literals come from **server-side** embedding of user text, not concatenated SQL from raw input.
- **YAML:** **`yaml.safe_load`** (`src/rag/config_loader.py`).
- **`scripts/migrate.py`** does not invoke shells with user-controlled strings.

**Residual risks**

- **`POST /ingest/chunks`** batch size is **capped** by **`MAX_INGEST_CHUNKS_PER_REQUEST`** (default **500**, HTTP **413**) — reduces memory/DB abuse vs unbounded batches; large **`content`** strings per row can still stress resources unless further bounded.

**Mitigations**

- Add Pydantic **`max_length`** on text fields if you need tighter bounds.

---

### A06:2025 — Insecure Design — **high (by intent for a demo)**

**Findings**

- **Global in-memory tuner / telemetry** — callers affect **shared** process state.
- **Demo embeddings** are deterministic hashes — **not** a security boundary for semantic secrecy.
- **SSRF:** The app does **not** fetch user-supplied URLs today.

**Watchouts**

- Future features such as **“fetch URL and embed”** need strict URL validation (scheme/host allowlists, block RFC1918/metadata URLs).

**Mitigations**

- Separate **admin/tuning** plane from **data plane** in production; **quotas**, **auditing**, real identity model.

---

### A07:2025 — Authentication Failures — **critical gap when API key unset (if network-exposed)**

**Findings**

- With **`RAG_API_KEY`** / **`API_KEY`** unset (demo default), there are **no sessions or per-user credentials** — full anonymous access except probe routes.
- With API key **set**, middleware accepts **`Authorization: Bearer`** or **`X-API-Key`** matching the shared secret — **no OAuth2**, **no MFA**, **no rotation story** in-app.

**Mitigations**

- Prefer **gateway-managed OAuth2/JWT**, **mTLS**, or **short-lived tokens** for production; treat the env-based shared secret as a **minimal** stub only.

---

### A08:2025 — Software or Data Integrity Failures — **low in-repo**

**Findings**

- No unsigned auto-update or arbitrary deserialization paths in core handlers.

**Mitigations**

- Sign releases / pin image digests; extend CI with container scanners when you publish images.

---

### A09:2025 — Security Logging and Alerting Failures — **medium**

**Findings**

- **Operational telemetry** exists for ingest/retrieve (`src/rag/telemetry.py`).
- **401** responses from optional API-key middleware give a **minimal** auth signal but **no structured security audit stream** or SIEM integration.

**Mitigations**

- Log **identity**, **policy decisions**, and **anomalies** to your logging stack.

---

### A10:2025 — Mishandling of Exceptional Conditions — **medium**

**Findings**

- **`POST /ingest/chunks`** maps database failures to **`HTTPException(detail=str(exc))`** — may **leak internal errors** (driver messages, constraint names) to clients (`src/rag/main.py`).
- **`GET /ready`** returns **`detail` strings** on failure paths for operators—typically lower risk than authenticated multi-tenant APIs but still worth generic messages if exposed broadly.

**Mitigations**

- Return **generic** messages to clients; log **`exc`** server-side only.

---

### Additional: Availability / abuse

- **`POST /ingest/chunks`** batch length is **capped** (see **`MAX_INGEST_CHUNKS_PER_REQUEST`**); **`POST /retrieve`** caps **`k` ≤ 200**.
- **No global rate limit** in-app — add at proxy or middleware if the API is reachable by untrusted clients.

---

## Quick hardening checklist (before any public deploy)

1. **Authentication + authorization** stronger than a single optional shared secret; **require** tenant context where multi-tenant.
2. **TLS** and **private networking** for API and database.
3. **Remove or restrict `/docs`**; **generic error messages** to clients (**A10**).
4. **Secrets** out of git; **strong** DB credentials; Postgres **not** on the public internet.
5. **Rate limits** and **request/body size limits** (proxy + app).
6. **Dependency scanning** (CI **`pip-audit`**) + **static analysis** (**Bandit**) + image scanning on release (**Trivy**/similar).
7. **Protect tuning/admin** endpoints from untrusted callers.
8. **`GET /ready`** for readiness only — combine with auth and network controls, not instead of them.
