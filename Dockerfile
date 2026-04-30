# API + deps; use with docker-compose (postgres service) for a full local stack.
FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY migrations ./migrations
COPY docker/entrypoint.sh /entrypoint.sh

RUN uv sync --frozen --no-dev \
    && chmod +x /entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"
# Overridden by docker-compose to reach the `postgres` service.
ENV DATABASE_URL=postgresql://rag:rag@postgres:5432/rag

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
