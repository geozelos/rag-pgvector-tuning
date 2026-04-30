#!/bin/sh
set -e
cd /app
python scripts/migrate.py
exec uvicorn rag.main:app --host 0.0.0.0 --port 8000 --app-dir src
