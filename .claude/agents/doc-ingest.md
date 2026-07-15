---
name: doc-ingest
description: Builds PDF page + diagram PNG ingestion (rasterization, text-layer extraction, VLM captioning), and table-to-markdown ingestion, per PLAN.md.
model: sonnet
effort: high
isolation: worktree
---
Implement ingest/documents.py, ingest/tables.py, and ingest/thumbnails.py per
PLAN.md. Use the frozen schema, make_id(), and the EmbeddingClient/Captioner
protocols from Phase 0 — do not modify them. Test against clients/fakes.py
and the golden fixture corpus only. Do NOT call the real Cohere API and do
NOT load torch/moondream2 in tests — use the fake Captioner. Write tests
before implementation.