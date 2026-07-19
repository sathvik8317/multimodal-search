---
name: retrieval-api
description: Builds RRF fusion, Cohere Rerank v3 integration, and the FastAPI + UI surface per PLAN.md.
model: sonnet
effort: high
isolation: worktree
---
Implement retrieve/fusion.py, retrieve/pipeline.py, api/main.py, and
api/static/index.html per PLAN.md. RRF must be rank-based (score-agnostic).
Rerank failure must degrade gracefully to RRF order with a logged warning,
not crash. Build and test against the frozen search() signature and a fake
table seeded from the golden fixture corpus. Do not call the real Cohere
API in tests.