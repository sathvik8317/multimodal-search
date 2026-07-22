# Embedding Migration: Cohere → OpenAI (split text encoder)

## Context

All embedding used to run through **Cohere Embed v4** into a **single 1536-d vector column**, and the
query was embedded once (Cohere text) and searched against it. That worked only because Embed v4 is a
*unified multimodal space*: a text-query vector is directly comparable to an image-doc vector.

Moving table/code text embedding to **OpenAI `text-embedding-3-small`** (cost) creates a **second,
incomparable vector space** — OpenAI 1536-d and Cohere 1536-d share a dimension count by coincidence
only; a query vector may be compared **only within its own space**. This forces a two-vector-retriever
design (independently of any caption decision) and is why the pipeline splits.

Clean split after this change: **OpenAI = the text encoder, Cohere = the image encoder.** Cohere Embed v4
stays for pdf_page + diagram *image* embedding (no substitute) and for Rerank v3.5. moondream2 captioning
is unchanged.

### Decisions (resolved in brainstorming)
- **Q1 = A:** caption rows (diagrams + scanned pdf_pages) keep their Cohere image vector **and** gain a
  second OpenAI vector over the VLM caption text. Strictly more recall (image + dense-caption + FTS),
  marginal cost only (the OpenAI column already exists for table/code).
- **Q2 = one table, two nullable vector columns.** One FTS index over `content_text` serves all rows.
  Every search embeds the query through **both** providers (Cohere + OpenAI) plus FTS → **three-way RRF**
  → Cohere rerank. Always both providers, not modality-routed (a second short-query embed is ~$0.00002;
  routing would trade recall for nothing).
- **Q3:** `OPENAI_API_KEY` added to the typed `Settings` model, mirroring `cohere_api_key`.
- **Q4:** full re-ingest + full eval re-run are **in scope here**, not a follow-up.

### Which vector(s) each row gets
| modality / case            | vector_cohere (image, Embed v4) | vector_openai (text, 3-small) | content_text (FTS) |
|----------------------------|:-------------------------------:|:-----------------------------:|--------------------|
| pdf_page, text layer       | ✅ page raster                  | —                             | text layer         |
| pdf_page, scanned (caption)| ✅ page raster                  | ✅ VLM caption                | VLM caption        |
| diagram                    | ✅ image                        | ✅ VLM caption                | VLM caption        |
| table                      | —                               | ✅ markdown                   | markdown           |
| code                       | —                               | ✅ context+body               | context+body       |

## Changes

### 1. Schema — `src/mmsearch/schema.py`
- `Row`: replaced `vector: list[float]` with `vector_cohere: list[float] | None = None` and
  `vector_openai: list[float] | None = None`.
- `__post_init__`: requires **at least one** vector set; validates `len(vector_cohere) ==
  COHERE_EMBED_DIM` and `len(vector_openai) == OPENAI_EMBED_DIM` when present; keeps the `content_text`
  non-empty check.
- `ARROW_SCHEMA`: two nullable `pa.list_(pa.float32(), <dim>)` fields (`vector_cohere`, `vector_openai`).
- `rows_to_table`: appends `None` for an absent vector (nullable fixed-size-list accepts `None`).

### 2. Config — `src/mmsearch/config.py`
- Replaced `EMBED_DIM`/`EMBED_MODEL` with per-provider constants:
  `COHERE_EMBED_MODEL="embed-v4.0"`, `COHERE_EMBED_DIM=1536`,
  `OPENAI_EMBED_MODEL="text-embedding-3-small"`, `OPENAI_EMBED_DIM=1536`. `RERANK_MODEL` unchanged.
- `RRF_K`: kept the value but rewrote the comment — its `k=20` tuning was against the *old single-space*
  index and is invalidated; retune against the re-ingested index (step 8).

### 3. Settings — `src/mmsearch/settings.py`
- Added `openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")`, mirroring
  `cohere_api_key` (unenforced here; the client raises lazily). No gate/dev-key policy change.

### 4. New client — `src/mmsearch/clients/openai.py`
- `OpenAIClient` implementing the existing `EmbeddingClient` protocol (`embed_documents`, `embed_query`).
- Lazy SDK build from `Settings().openai_api_key`, same shape as `CohereClient._build_default_sdk`
  (raises `RuntimeError` only when the key is actually needed; `sdk=` injectable for tests).
- **Text only:** `embed_documents` raises `ValueError` if any `EmbedInput.image_bytes` is set.
- No `input_type` (text-embedding-3 has no search_query/search_document distinction — unlike Cohere).
- Reuses the retry/backoff pattern from `cohere.py` against real `openai` transient errors
  (`RateLimitError`, `APITimeoutError`, `InternalServerError` — names/hierarchy confirmed against the
  installed `openai>=1.0` SDK, not assumed from memory); batches inputs (constant, 96).

### 5. Ingest routing — `src/mmsearch/ingest/*`
- Added `Embedders(image: EmbeddingClient, text: EmbeddingClient)` (frozen dataclass, in
  `clients/protocols.py`). Threaded through `ingest_corpus` → `_ingest_one` → `ingest_pdf`/`ingest_diagram`
  (both need image + text); `ingest_table`/`ingest_code_file` still take a single `EmbeddingClient` (the
  text embedder only, passed as `embedders.text`) since they never need an image embed.
- `documents.py`: `ingest_pdf`/`ingest_diagram` embed the image via `embedders.image` → `vector_cohere`;
  for `VLM_CAPTION` rows, also embed the already-computed caption via `embedders.text` → `vector_openai`.
- `tables.py`/`code.py`: embed via the passed text embedder → `vector_openai` (was Cohere text).
- `ingest/cli.py`: builds both `CohereClient()` and `OpenAIClient()`, wraps in `Embedders`, passes through.

### 6. Pipeline — `src/mmsearch/retrieve/pipeline.py`
- `build_search_fn(table, cohere_embedder, openai_embedder, reranker, *, mode, ...)`.
- Per search: `cohere_query_vector = cohere_embedder.embed_query(q)` → `table.search(cohere_query_vector,
  query_type="vector", vector_column_name="vector_cohere").limit(fetch_n)`; same for OpenAI on
  `vector_openai`; FTS as before.
- Three-way RRF: `reciprocal_rank_fusion([cohere_ids, openai_ids, fts_ids], k=rrf_k)` (fusion.py already
  took N lists — no change needed there). Rerank shortlist unchanged.
- Query fan-out is sequential (smaller diff), marked `# ponytail: sequential query embeds; wrap the two
  network calls in a ThreadPool(2) if search latency matters`.
- Modes: `vector-only` now fuses both vector retrievers (Cohere ∪ OpenAI, two-way RRF, no FTS);
  `rrf-only`/`rrf+rerank` are three-way (Cohere + OpenAI + FTS).

### 7. Wiring & deps
- `src/mmsearch/api/server.py`: builds both clients; `build_search_fn(_table, _cohere_client,
  _openai_client, _cohere_client, mode="rrf+rerank")` (Cohere client is still the reranker).
- `pyproject.toml`: added `openai>=1.0` to core `dependencies`.
- Tests: updated fakes/wiring for two vector columns + two embedders across `test_pipeline.py`,
  `test_cohere_client.py`, the new `test_openai_client.py`, ingest tests, schema/config/settings tests.
  `OpenAIClient` has a `demo()` self-check (asserts image input is rejected), runnable via
  `python -m mmsearch.clients.openai`.

### 8. Re-ingest + re-eval (Q4, in scope) — **not yet run**
- Schema change is incompatible with the existing table (`merge_insert` can't add columns to an old
  table) → drop `data/lancedb`, then `mmsearch ingest <corpus>` to rebuild all rows against the new
  pipeline.
- `labels.yaml` ids are embedding-independent → unchanged, re-run (not re-written); re-validate ids
  against the fresh index.
- Re-run eval ablations (`rrf-only`, `rrf+rerank`, per-retriever isolation) and re-tune `RRF_K` on the
  new two-space index; update the `config.py` comment and `README.md`'s eval table with the new evidence.

## Verification

1. **Foundational assumptions — both confirmed before writing implementation code:**
   - **LanceDB nullable multi-vector:** confirmed live — a 2-column table with one column `None` per row
     stores/retrieves correctly; `table.search(qv, vector_column_name="vector_cohere")` returns only rows
     with that column populated; `merge_insert` handles nulls without error.
   - **OpenAI embeddings API (live):** called the real `text-embedding-3-small` endpoint; the returned
     vector's length is exactly `1536`, matching `OPENAI_EMBED_DIM`. Inspected the installed `openai>=1.0`
     SDK's exception hierarchy directly (not from memory): `RateLimitError` → `APIStatusError` →
     `APIError` → `OpenAIError`; `APITimeoutError` → `APIConnectionError` → `APIError`; `InternalServerError`
     → `APIStatusError` → `APIError`. All three names used in `OpenAIClient._TRANSIENT_ERRORS` are correct.
2. `pytest` green (unit suite, `-m "not live"`): **296 passed.**
3. `mmsearch ingest <corpus>` rebuilds ~113 rows; report shows table/code with `vector_openai`,
   diagrams/scanned pages with both. **Pending** (step 8).
4. Eval ablation table prints; aggregate hit-rate ≥ pre-migration baseline; record the retuned `RRF_K`.
   **Pending** (step 8).
5. `uvicorn mmsearch.api.server:app` + a `/search` call returns fused results across modalities.
   **Pending** (needs the rebuilt index).
6. Optional `live`-marked smoke tests for `OpenAIClient` against the real API (opt-in, like Cohere's).
   **Not built** — the one-off live check in step 1 above served the same purpose for this migration.

## Known gaps

**No failure protection on either provider's query embed call — unlike the reranker.** In
`pipeline.py`'s `search()` closure, `cohere_embedder.embed_query(query)` and
`openai_embedder.embed_query(query)` are both called with no `try`/`except` around them. Contrast with
the reranker call a few lines later, which is wrapped and degrades gracefully to RRF-fused order on any
exception. There is no equivalent degradation path for a failed query embed on either provider, and
`api/main.py`'s `/search` handler doesn't catch anything either — so once a provider's own retry/backoff
is exhausted (or the failure is non-transient), the exception propagates unhandled and surfaces as a loud
500, not a degraded-but-working response.

This is a **real regression versus pre-migration**, not just a pre-existing gap carried forward: a query
embed used to depend on one provider (Cohere) succeeding; it now depends on **two** providers succeeding
independently, which doubles the availability surface area for a search request to fail outright.

Not blocking this merge, but worth a future fix — e.g. catch a failed embed on one provider and degrade
to single-provider vector retrieval (skip that provider's `table.search(...)` call, fuse whatever's left),
mirroring the reranker's existing fallback pattern rather than inventing a new one.

## Out of scope (future levers, noted not built)
- **rrf-only default** to drop Cohere Rerank spend entirely.
- **Modality-scoped query routing** to skip a provider's query embed when a modality is filtered out.
- Embedding pdf **text-layer** pages through OpenAI as an extra dense signal (only captions get it now).
