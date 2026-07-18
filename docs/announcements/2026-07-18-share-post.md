# Share posts — copy/paste

Use these after the repo is pushed with the measured chart in the README.

---

## Show HN (Hacker News)

**Title:**

```text
Show HN: Lab to see how pgvector HNSW ef_search trades latency for recall
```

**Text:**

```text
Most RAG tutorials stop at “call the LLM.” In production with Postgres + pgvector, the pain is often retrieval: HNSW ef_search (and IVFFlat probes) change latency and recall, but teams guess the knobs.

This is a small open-source lab (FastAPI + Docker, no embedding API keys) that measures it:

- seed a topical 10k corpus
- rebuild a modest HNSW so the knob is visible
- sweep ef_search and plot p50 latency vs recall@k against an exact (seq-scan) oracle

Example from a local run (your numbers will differ):

    ef    p50_ms   recall
     8     0.84    0.067
    16     0.85    0.067
    40     0.88    0.067
    96     0.89    0.067
   200     1.13    0.533

Repo: https://github.com/geozelos/rag-pgvector-tuning

    make up && make demo

Not a chatbot framework — just inspectable retrieve + tuning so you can see the tradeoff yourself.
```

---

## Short (X / LinkedIn / Discord)

```text
pgvector RAG tip: ef_search is a latency↔recall knob, not a magic default.

I open-sourced a tiny lab that measures it on Postgres (Docker, no API keys):

make up && make demo
→ ASCII chart of p50 latency vs recall@k vs exact oracle

https://github.com/geozelos/rag-pgvector-tuning
```

---

## Where to post (order)

1. **Hacker News** — Show HN (weekend mornings US often quieter; weekdays ~8–10am ET competitive)
2. **r/PostgreSQL** + **r/MachineLearning** (link post + 2–3 sentence context)
3. **pgvector / Postgres Discord or Slack** if you’re in one — paste the short version + chart
4. LinkedIn — short version + screenshot of the README chart

Do not spam identical walls of text; adapt one sentence to the community.
