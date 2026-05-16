# Recipe: Load testing and tuner

**Goal:** Generate steady **`POST /retrieve`** traffic so in-process telemetry has enough samples for **`/tuner/recommend`**.

## Why

The MVP tuner reads rolling latency percentiles from memory. A handful of manual clicks rarely fills that window; a paced script is easier.

## Command

```bash
uv sync
uv run python scripts/load_retrieve_qps.py --qps 15 --duration 45 --tenant-id demo
```

With API key enforced:

```bash
uv run python scripts/load_retrieve_qps.py --api-key "$RAG_API_KEY" --qps 10 --duration 30 --tenant-id demo
```

JSON summary:

```bash
uv run python scripts/load_retrieve_qps.py --qps 20 --duration 20 --json
```

## Then

```bash
curl -s -X POST http://127.0.0.1:8000/tuner/recommend
curl -s -X POST "http://127.0.0.1:8000/tuner/step?auto_apply=false"
```

Understand **`config/tuner_guardrails.yaml`** before using **`auto_apply=true`**.

## Note

With **`EMBEDDING_BACKEND=demo`**, semantic quality is weak; **latency** is the clearest signal for knob experiments.

See **`python scripts/load_retrieve_qps.py --help`** for `--base-url`, `--query`, `--k`, concurrency, etc.
