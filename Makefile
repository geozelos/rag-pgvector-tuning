# Operator shortcuts for rag-pgvector-tuning.
#
# Compose CLI: use V2 when available. If `docker compose` fails on your machine, run:
#   make up COMPOSE=docker-compose
#
COMPOSE ?= docker compose
export DATABASE_URL ?= postgresql://rag:rag@localhost:5433/rag

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available targets
	@echo "rag-pgvector-tuning — common tasks"
	@echo ""
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## ' $(MAKEFILE_LIST) | sort | sed 's/Makefile://' | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: up
up: ## Start Postgres + API (build images)
	$(COMPOSE) up --build -d

.PHONY: down
down: ## Stop and remove containers
	$(COMPOSE) down

.PHONY: ps
ps: ## Show compose service status
	$(COMPOSE) ps

.PHONY: logs
logs: ## Follow API container logs
	$(COMPOSE) logs -f api

.PHONY: postgres-only
postgres-only: ## Start only Postgres (Option B: API on host)
	$(COMPOSE) up -d postgres

.PHONY: migrate
migrate: ## Apply SQL migrations (uses DATABASE_URL)
	uv run python scripts/migrate.py

.PHONY: test
test: ## Run full pytest with coverage (integration skips if DB down)
	uv run pytest tests/ -q --cov=rag --cov-branch

.PHONY: test-unit
test-unit: ## Run pytest without integration tests
	uv run pytest tests/ -q --cov=rag --cov-branch -m "not integration"

.PHONY: integration
integration: ## Run PostgreSQL integration test (needs Postgres on DATABASE_URL)
	uv run pytest tests/test_integration_api.py -m integration -v --no-cov

.PHONY: security
security: ## Bandit + pip-audit (matches CI security job shape)
	uv run bandit -r src/rag scripts -ll -c pyproject.toml
	uv export --frozen --no-dev --no-emit-project --no-hashes -o .deps-audit.txt
	uv run pip-audit -r .deps-audit.txt
	@rm -f .deps-audit.txt

.PHONY: openapi-snapshot
openapi-snapshot: ## Regenerate tests/fixtures/openapi.json after API schema changes
	UPDATE_OPENAPI_SNAPSHOT=1 uv run pytest tests/test_openapi_contract.py -q
