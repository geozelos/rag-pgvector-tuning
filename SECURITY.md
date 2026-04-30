# Security

This repository is a **learning and demonstration** project. Treat the findings below as a **threat model and hardening checklist**, not as certification.

If you discover a security issue in this codebase, please report it **privately** to the maintainer when sensitive, or open a public issue for non-sensitive items (documentation-only, etc.).

---

## OWASP Top 10 (2021) — review of this codebase

Mappings are **qualitative** (code inspection). Run **`pip-audit`**, **Dependabot**, or **`uv lock` + OSV** regularly for **known vulnerable dependencies** (A06); results change over time.

### A01: Broken Access Control — **high (if exposed beyond localhost)**

**Findings**

- **No authentication or authorization** on any HTTP route (`src/rag/main.py`). Any client that can reach the API can:
  - **Ingest or overwrite** chunks (global corpus), including other tenants’ `(doc_id, chunk_index)` keys.
  - **Retrieve** vectors and text. **`tenant_id` / `source_type` are optional filters** — omitting them returns hits across **all** rows (`/retrieve` builds `WHERE` only when filters are set).
  - **Change global runtime tuning** (`PATCH /config/runtime-search`), **clear telemetry state**, **drive the tuner** (`/tuner/*`).
- **OpenAPI `/docs` and `/redoc`** expose the full attack surface and simplify abuse.
- There is **no per-tenant API key, JWT, or RBAC**.

**Mitigations (production-oriented)**

- Put the API **behind an API gateway / reverse proxy** with **mTLS or OAuth2/JWT**, **per-route scopes**, and **network policies** (only trusted callers).
- Enforce **tenant isolation** in the application layer (required tenant context + server-side checks, or row-level security in PostgreSQL).
- **Disable or protect** `/docs` in production (or serve only on an internal network).

---

### A02: Cryptographic Failures — **medium (deployment-dependent)**

**Findings**

- **`DATABASE_URL`** may contain **plain-text credentials** (`.env`, Docker Compose env). Default compose uses a **weak, documented password** (`rag` / `rag`) — fine for local dev, **not** for the internet.
- **No TLS** is configured in the application; HTTPS is assumed to be **terminated** elsewhere (reverse proxy / load balancer) if needed.
- **No encryption-at-rest** is configured here (rely on disk encryption / managed DB).

**Mitigations**

- Use **secrets management** (env from vault, not committed files); **rotate** DB passwords.
- Terminate **TLS** at the edge; use **`sslmode=require`/`verify-full`** for Postgres client connections where supported.

---

### A03: Injection — **lower for SQL; other inputs still matter**

**Findings**

- **SQL:** Ingest uses **parameterized** `executemany` with `$1…$6` placeholders. Retrieve builds a **fixed** `WHERE` template and passes filter values as **bound parameters** (`$3`, …) — **not** string-concatenated user text into SQL values. **Vector literal** for search is derived from server-side embedding of `query`, not raw user SQL.
- **YAML:** Config uses **`yaml.safe_load`** (`src/rag/config_loader.py`) — avoids arbitrary object construction from YAML.
- **Command execution:** `scripts/migrate.py` does not shell out with user input.

**Residual risks**

- Very large **ingest batches** (unbounded list length in `IngestPayload`) can stress **memory and DB** (availability / DoS), not classic SQLi.

**Mitigations**

- Cap **`chunks` length and row field sizes** (Pydantic `max_length`, max batch size).

---

### A04: Insecure Design — **high (by intent for a demo)**

**Findings**

- **Global in-memory tuner / telemetry** — any caller affects **shared** process state.
- **Multi-tenancy is client-declared** (`tenant_id` on ingest/filters), not cryptographically enforced.
- **Demo embeddings** are deterministic hashes — **not** a security boundary; they are for **learning**, not production semantic search.

**Mitigations**

- Redesign for production: **authn/authz**, **quotas**, **auditing**, **separate tuning admin API** from data plane.

---

### A05: Security Misconfiguration — **high (default stack)**

**Findings**

- **Postgres published** on host port **5433** in `docker-compose.yml` — broad **localhost** exposure; on misconfigured hosts this can widen risk.
- **Default credentials** and **debug-grade** logging (e.g. ingest failures may return **`str(exc)`** to the client in `HTTPException`) can **leak internal errors**.
- **No security headers, rate limiting, or CORS policy** defined in code (FastAPI defaults only).

**Mitigations**

- **Bind DB** to internal networks only in real deployments; **do not** expose Postgres publicly.
- Map **exceptions** to **generic** client messages; log details **server-side** only.
- Add **rate limiting** (proxy or middleware), **security headers**, and explicit **CORS** allowlists if browser clients exist.

---

### A06: Vulnerable and Outdated Components — **ongoing**

**Findings**

- Dependencies are pinned in **`uv.lock`**; they can still have **disclosed CVEs** at any time.

**Mitigations**

- Run regularly, e.g.:

  ```bash
  uv tool run pip-audit -r <(uv export --no-dev --frozen)
  ```

  Or enable **Dependabot / Renovate** on the repository. **Rebuild** images after upgrades.

---

### A07: Identification and Authentication Failures — **critical gap (if network-exposed)**

**Findings**

- **No login, sessions, or API keys.** Every endpoint is **anonymous full access**.

**Mitigations**

- Add **OAuth2**, **API keys**, or **mTLS**; validate **every mutating and sensitive read** path.

---

### A08: Software and Data Integrity Failures — **low in-repo**

**Findings**

- No auto-update or unsigned deserialization paths in the app. **Docker images** should be built from **pinned base images** and scanned in CI (**Trivy**, **Grype**, etc.).

**Mitigations**

- Sign images / use digest pins; verify supply chain in CI.

---

### A09: Security Logging and Monitoring Failures — **medium**

**Findings**

- **Structured telemetry** exists for ingest/retrieve (**good for ops**), but there is **no security event stream** (auth failures, policy denials), and **no centralized SIEM** integration.

**Mitigations**

- Log **client identity**, **decisions**, and **anomalies**; forward to your logging platform.

---

### A10: Server-Side Request Forgery (SSRF) — **low**

**Findings**

- The app does **not** fetch user-supplied URLs in `src/rag/`. **No obvious SSRF** surface in the current retrieval/ingest path.

**Watchouts**

- If you later add **“fetch URL and embed”**, validate schemes/hosts and **block** internal networks.

---

### Additional: Availability / abuse

- **`POST /ingest/chunks`** has **no max batch**; large payloads can cause **CPU/memory** pressure (embedding + DB).
- **`POST /retrieve`** caps **`k` ≤ 200** — reasonable.
- **No global rate limit** — brute force or corpus scraping is trivial if the API is reachable.

---

## Quick hardening checklist (before any public deploy)

1. **Authentication + authorization** on all routes; **require** tenant context where multi-tenant.
2. **TLS** and **private networking** for API and database.
3. **Remove or restrict `/docs`**, **generic error messages** to clients.
4. **Secrets** out of git; **strong** DB credentials; Postgres **not** on the public internet.
5. **Rate limits** and **request/body size limits** (proxy + app).
6. **Dependency + image scanning** on every release.
7. **Disable** or protect **tuning/admin** endpoints from untrusted callers.
