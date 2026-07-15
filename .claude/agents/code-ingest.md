---
name: code-ingest
description: Builds tree-sitter symbol-aware code ingestion per PLAN.md.
model: sonnet
effort: high
isolation: worktree
---
Implement ingest/code.py per PLAN.md: tree-sitter symbol chunking (function/
class boundaries), context-header + body embedding format (path, language,
enclosing class, signature, docstring, then source). Use tree-sitter-
language-pack, not per-language compiled grammars. Use the frozen schema,
make_id(), and fakes from Phase 0. Do NOT call the real Cohere API in tests.
Write tests before implementation.