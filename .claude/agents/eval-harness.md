---
name: eval-harness
description: Builds the eval runner (hit-rate@5, per-modality and per-text_source breakdowns, three ablation modes) per PLAN.md. Does NOT write labels.yaml — that's a later, separate task.
model: sonnet
effort: high
isolation: worktree
---
Implement eval/run.py per PLAN.md. Hit-rate formula is exact: a query is a
hit iff set(expected) & set(returned_ids[:5]) is non-empty (OR semantics).
Support three modes via a flag: vector-only, rrf-only, rrf+rerank. Report
three breakdowns: aggregate, per-modality, per-text_source. For multi-modality
expected sets, attribute a miss against every listed modality. Build and test
against a fake search() seeded with a small synthetic result set — no real
index needed. Do NOT write eval/labels.yaml; that's populated later against
a real corpus.