# Multimodal Search for Engineers — Design & Build Plan

_A unified search index over PDFs, diagrams, tables, and code for technical teams.
Portfolio SaaS demo — spirit of Morphik/Reducto, scoped to a single-user demo._

Status: **design approved, pre-implementation.** Do not implement past Phase 0 until
the user has ingested a real corpus and reviewed this plan.

---

## 1. Goal & scope

Single embedded search system that unifies four modalities into **one LanceDB table**
and one embedding space, retrieved by hybrid (vector + full-text) fusion and a
cross-encoder reranker, with an **eval harness built before ranking is tuned**.

**In scope:** ingest (PDF pages, standalone diagram images, standalone tables, code),
hybrid retrieval + rerank, a FastAPI JSON API, a minimal single-page web UI that renders
result cards with thumbnails, and a hit-rate@5 eval harness with ablation modes.

**Out of scope (YAGNI):** multi-tenancy, auth, an external vector DB server, OCR,
PDF-region/layout detection, late-interaction (ColPali-style multi-vector) retrieval,
incremental/streaming ingest, horizontal scale.

### Decisions locked during brainstorming

| # | Decision | Rationale |
|---|---|---|
| 1 | **One embedding space, one table.** Cohere Embed v4 for all modalities. | Keeps query = one embed call; scores comparable within a modality. Escalation ladder held in reserve (§7). |
| 2 | **Diagrams/tables are standalone files, not PDF regions.** | No layout-detection model. Four clean file-type-keyed ingestion paths. |
| 3 | **PDFs & diagrams embedded as page images (ColPali-style, single-vector).** But **text is still extracted for FTS + rerank.** | Page-as-image keeps layout/figure signal; text layer / VLM caption keeps the reranker honest. |
| 4 | **Tables are NOT rasterized.** Serialize to markdown, embed as text. | A CSV has no visual layout to preserve; rasterizing it would add a headless-browser dependency for nothing. |
| 5 | **No row ever has empty `content_text`.** PDF text layer → table markdown → code source → **VLM caption** for text-less images (diagram PNGs, scanned pages). | FTS and Rerank v3 are text-only. Empty text silently degrades RRF to vector-only and passes empty strings to the reranker. |
| 6 | **Captioner = local VLM (moondream2) on the RTX 3050**, behind a `Captioner` protocol. | Zero marginal cost, offline, ingest is batch work so latency is irrelevant. Swappable via the protocol. |
| 7 | **Eval labels are hand-written (~25 pairs).** | Only phrasing that isn't leaked from the source doc gives an honest hit-rate@5. |
| 8 | **Corpus is small (~200–500 rows) → brute-force vector scan, no ANN index.** | No recall/latency approximation error; hit-rate@5 measures ranking, not index config. Index type is a one-line config switch for later. |
| 9 | **Demo surface = FastAPI + minimal static web UI** with thumbnail result cards. | Makes *multimodal* legible: a diagram returned for a text query is the money shot. |

### External dependencies

- **Cohere API** (`COHERE_API_KEY`): Embed v4 (`input_type` = `search_document` / `search_query`),
  Rerank v3. Both behind protocols with fakes.
- **moondream2** via `transformers` + `torch` (CUDA). **Lazy-imported** — only the doc
  ingester loads torch; the API and retrieval path never do.
- **tree-sitter** via `tree-sitter-language-pack` (prebuilt grammars, no per-language
  compilation on Windows).
- **PyMuPDF** (`fitz`): PDF page rasterization + text-layer extraction + thumbnails.

---

## 2. Finalized LanceDB schema

**One table, `chunks`.** Arrow schema:

| column | Arrow type | notes |
|---|---|---|
| `id` | `string` | deterministic **and human-readable** (see §2.1). Primary key for upsert. |
| `modality` | `string` | `pdf_page` \| `diagram` \| `table` \| `code`. Top-level so LanceDB can filter on it. |
| `content_text` | `string` | **FTS-indexed. Never empty** (Decision 5). |
| `text_source` | `string` | `pdf_text_layer` \| `vlm_caption` \| `table_markdown` \| `code_source`. Not redundant with `modality` — see §2.2. |
| `vector` | `fixed_size_list<float32>[1536]` | Cohere Embed v4 output dim. Single vector per row. |
| `source_path` | `string` | corpus-relative path to the original source file. |
| `thumbnail_ref` | `string` | path **relative to `THUMBNAILS_DIR`**; `""` for `code` and `table`. |
| `metadata` | `string` | **JSON string**, per-modality fields (see §2.3). |

### 2.1 `id` scheme (`make_id` in `schema.py`)

Human-readable **because** eval labels are hand-typed (Decision 7). Deterministic so
re-ingest is idempotent (upsert on `id`).

| modality | format | example |
|---|---|---|
| pdf_page | `pdf:{relpath}#p{page_no}` | `pdf:specs/rfc.pdf#p14` |
| diagram | `img:{relpath}` | `img:docs/auth-flow.png` |
| table | `tbl:{relpath}` | `tbl:data/latency.csv` |
| code | `code:{relpath}#{qualname}` | `code:src/ingest/base.py#PdfIngester.rasterize` |

`{relpath}` is always corpus-root-relative and POSIX-separated (stable across OSes).

### 2.2 Why `text_source` is not redundant with `modality`

1:1 for three modalities, but **`pdf_page` maps to two sources**: `pdf_text_layer`
normally, `vlm_caption` when the page is scanned (`get_text()` returns near-nothing →
fall back to the captioner). That single exception lets the eval harness answer
"are VLM captions pulling their weight, or is all recall coming from text layers?"
before any ranking is touched.

### 2.3 `metadata` JSON contents (per modality)

JSON string, not an Arrow struct — per-modality fields genuinely differ and a struct
would force a ~10-column mostly-null union onto every row. The one field worth filtering
on (`modality`) is already top-level.

- **pdf_page**: `{ "page_no": int, "n_pages": int }`
- **diagram**: `{ "width": int, "height": int, "caption_model": str }`
- **table**: `{ "n_rows": int, "n_cols": int, "columns": [str, ...] }`
- **code**: `{ "lang": str, "qualname": str, "kind": "function"|"class"|"method", "start_line": int, "end_line": int }`

---

## 3. Module structure

`← P0` = Phase-0 frozen contract (built solo, blocks all parallel work).

```
multimodal-search/
  pyproject.toml
  .env.example                       # COHERE_API_KEY
  README.md
  src/mmsearch/
    config.py            ← P0   paths, EMBED_DIM, RRF_K, FETCH_N, RERANK_M, TOP_K, model ids
    schema.py            ← P0   Arrow schema, Modality enum, Row dataclass, make_id()
    db.py                ← P0   open_table(), ensure_fts_index(), upsert(rows)
    clients/
      protocols.py       ← P0   EmbeddingClient | Reranker | Captioner (typing.Protocol)
      cohere.py          ← P0   embed_documents / embed_query / rerank + retry + batching
      captioner_local.py        moondream2; lazy torch import; disk cache by image hash
      fakes.py           ← P0   deterministic fakes for all three protocols
    ingest/
      base.py                   Ingester protocol + shared _embed_and_write driver
      documents.py       (a)    PDF pages (rasterize + text-layer/VLM) + diagram PNGs
      tables.py          (a)    csv/xlsx → markdown → text embed
      thumbnails.py      (a)    write PNG thumbs into THUMBNAILS_DIR
      code.py            (b)    tree-sitter symbol-aware chunking + context header
      cli.py                    `mmsearch ingest <path>`
    retrieve/
      types.py           ← P0   SearchResult dataclass; search(query, k) signature
      fusion.py          (c)    reciprocal_rank_fusion(lists, k=RRF_K)
      pipeline.py        (c)    embed_query → vector+FTS → RRF → rerank → SearchResult
    api/
      main.py            (c)    GET /search, GET /healthz, static /thumbnails, static /ui
      static/index.html  (c)    single-page result-card UI (no framework)
    eval/
      dataset.py         ← P0   labels.yaml format + loader + validator
      labels.yaml        (d,Phase2)  ~25 hand-written query→expected pairs
      run.py             (d)    hit-rate@5, 3 breakdowns × 3 ablation modes
  tests/
    fixtures/corpus/     ← P0   golden fixture: 2 PDF pages, 1 diagram PNG, 1 csv, 1 .py
    ...                        per-component tests (see §6)
```

---

## 4. Retrieval pipeline

`search(query: str, k: int = TOP_K) -> list[SearchResult]`

```
query
  └─ embed_query            1 Cohere call, input_type=search_query
       ├─ vector search     table.search(qvec).limit(FETCH_N)            → list A (by cosine)
       └─ fts search        table.search(query, "content_text").limit(FETCH_N) → list B (by BM25)
            └─ RRF fuse(A, B, k=RRF_K)     rank-based, score-agnostic     → fused[:FETCH_N]
                 └─ Rerank v3 on fused[:RERANK_M].content_text           → reranked
                      └─ top-k SearchResult
```

**`SearchResult`**: `id, modality, score, snippet, thumbnail_ref, source_path, text_source`.

**Tunable config surface (the only knobs; set by eval evidence, not by feel):**
`FETCH_N=50` per retriever · `RERANK_M=25` fused docs reranked · `TOP_K=5` returned · `RRF_K=60`.

**Design points:**
- **RRF is score-blind** (`1/(RRF_K+rank)`) — never reconciles cosine vs BM25 scales.
  Both lists are always populated because `content_text` is never empty (Decision 5).
- **Rerank degradation is a designed path.** On rerank failure / missing key: log a
  warning, return RRF order. System stays up; results merely un-reranked. Eval can toggle
  rerank off to *measure its contribution*.
- **Snippet UX for images:** result card leads with the **thumbnail** (the real content);
  the VLM caption is shown as sub-text. The human sees the image; the machine ranked on words.

---

## 5. Eval harness

### 5.1 Label format (`labels.yaml`)

```yaml
- query: "what's the retry backoff on the ingest worker"
  expected: ["code:src/ingest/base.py#_embed_and_write"]
- query: "the diagram showing auth token flow"
  expected: ["img:docs/auth-flow.png"]
- query: "p99 latency numbers for the reranker"
  expected: ["tbl:data/latency.csv", "pdf:specs/bench.pdf#p3"]   # either is a hit (OR)
```

### 5.2 Hit-rate@5 semantics (definition of record)

`expected` is a **set of acceptable answers, combined with OR.** A query is a hit iff at
least one expected id appears in the top-5 returned ids:

```python
hit = len(set(expected) & set(returned_ids[:TOP_K])) > 0     # OR — any overlap = hit
hit_rate = num_queries_with_hit / num_queries                 # each query contributes 0 or 1
```

**Never AND. Never fractional.** The multi-id list exists for cases where two documents
legitimately answer one question (the latency example above): surfacing *either* is a win.

**Per-modality attribution rule (keeps denominators honest):** a query with a multi-modality
`expected` is credited to the modality of the id that actually hit; on a miss it counts
against **every** modality it lists. No double-crediting.

### 5.3 Three breakdowns × three ablation modes

`run.py` reports:

1. **Aggregate hit-rate@5** — the headline number.
2. **Per-modality** — the decision gate for §7 (split code embeddings or not).
   Catches `code: 0.40` hiding under `aggregate: 0.80`.
3. **Per-`text_source`** — isolates whether `vlm_caption` rows earn the torch dependency
   vs `pdf_text_layer`.

Ablation modes (one `--mode` flag): `vector-only` · `rrf-only` · `rrf+rerank`.

The resulting **3 modes × 3 breakdowns table is the portfolio artifact** — it turns
"I built a search thing" into "reranking added N points, concentrated in the diagram modality."

### 5.4 Two milestones, not one (do not conflate)

- **Harness code** (`run.py`, `dataset.py`): parallel-safe from day one — builds against a
  **fake `search()`** and needs no real index.
- **The 25 labels** (`labels.yaml`): a **Phase-2 gated task** — you cannot label ids that
  haven't been minted. Written only after a real corpus is ingested.

---

## 6. Build order, parallelization & worktree-safety

### Phase 0 — the contract (solo, blocks everything)

`git init` first (repo does not exist yet). Then freeze and land, tested green against
fakes: `config.py`, `schema.py` + `make_id`, `db.py`, all three `protocols.py`, `cohere.py`
client (real, but exercised via contract tests), `fakes.py`, `retrieve/types.py`
(`SearchResult` + `search()` signature), `eval/dataset.py` (label format), the thumbnail
path convention, and the **golden fixture corpus** under `tests/fixtures/corpus/`.

**Exit criteria:** CI green on fakes; every downstream track can import a stable, typed contract.

### Phase 1 — four parallel tracks (worktree-safe after P0)

| Track | Scope | Worktree-safe? | Coupling neutralized by P0 |
|---|---|---|---|
| **(a) doc ingest** | PDF pages, diagram PNGs, tables, thumbnails, VLM caption | ✅ | schema, `make_id`, `EmbeddingClient`/`Captioner`, thumbnail convention |
| **(b) code ingest** | tree-sitter symbol chunking + context header | ✅ | schema, `make_id`, `EmbeddingClient`; own isolated env dep (grammars) touches no one |
| **(c) retrieval+API+UI** | fusion, pipeline, FastAPI, static UI | ✅ | `SearchResult`+`search()` sig, `Reranker`; builds against a **fake table** from the fixture |
| **(d) eval harness (code)** | `run.py`, ablations, breakdowns | ✅ | label format + `search()` sig; builds against a **fake `search()`** |

**All four are genuinely independent — but only the *code* is.** In Phase 1 every track
builds/tests **against fakes only**: no real Cohere calls, no real GPU. This keeps the
parallel phase free of cost and serializes the two contended resources (Cohere credits,
the single 4GB GPU) to the one-time real ingest at the merge point.

**Not parallelizable (sequenced deliberately):**
- **The real index** — a+b+c must merge before a real search returns real rows (a merge-point).
- **The 25 labels** — Phase-2, gated on the real corpus existing.

### Phase 2 — integrate & tune (solo)

1. Merge a+b+c. Run **one** real ingest over the demo corpus (user-driven; spends credits + GPU once).
2. Hand-write the ~25 labels in `labels.yaml` against the now-minted real ids.
3. Run `run.py` across all 3 modes × 3 breakdowns.
4. Tune `FETCH_N` / `RERANK_M` / `RRF_K` on evidence.
5. **If** per-modality `code` hit-rate is below threshold, climb the §7 ladder.

---

## 7. Code-embedding escalation ladder (deferred, evidence-gated)

Unified space is the default (Decision 1). Splitting into a second code-specialized model
is expensive: two incomparable vector spaces, forced rank-based cross-retriever merge, two
embed calls per query. So escalate only if **per-modality code hit-rate** (§5.3) is bad,
in this order:

1. **Enrich the code embedding input.** Embed a context header (file path, language,
   enclosing class, signature, docstring) + body — not the raw body. ~10 lines; usually
   closes most of the gap.
2. **Confirm FTS is carrying code.** Code is where lexical search is strongest (identifiers
   are exact tokens). If code recall is low, verify list B is actually firing before blaming
   the vector half.
3. **Split** into a code-specialized model + cross-retriever rank merge — **last resort only.**

---

## 8. Test plan (per component)

**Global:** all Phase-1 tests run against `fakes.py` and the golden fixture — deterministic,
no network, no GPU. Real-API smoke tests are marked and opt-in (`-m live`).

**P0 — contract**
- `make_id` round-trips and is stable across OS separators; ids are unique per fixture row.
- `db.upsert` is idempotent (re-upsert same id → no duplicate row).
- FTS index is created and returns fixture rows for a known token.
- Fakes satisfy the protocols (structural typing check) and are deterministic.

**(a) doc ingest**
- PDF page → correct `id`, `page_no`, non-empty `content_text` from text layer, thumbnail written.
- Scanned-page path (text layer empty) → `text_source == vlm_caption`, caption non-empty (fake captioner).
- Diagram PNG → `img:` id, `vlm_caption`, thumbnail written, `thumbnail_ref` relative to `THUMBNAILS_DIR`.
- Table csv → markdown serialization, `n_rows`/`columns` in metadata, `thumbnail_ref == ""`.
- Caption disk cache: second ingest of same image does not invoke the captioner.

**(b) code ingest**
- Function / class / method boundaries chunked correctly on the fixture `.py` (no fixed-size splits).
- `qualname` and `code:...#qualname` id match; `start_line`/`end_line` correct.
- Context header present in embedded text; multi-language smoke via `tree-sitter-language-pack`.

**(c) retrieval + API + UI**
- `reciprocal_rank_fusion`: known A/B rankings → hand-computed fused order (score-blind).
- Pipeline against fake table: returns `TOP_K`, correct `SearchResult` shape.
- Rerank-failure path: reranker fake raises → pipeline returns RRF order + logs warning (no crash).
- API: `/search` JSON shape, `/healthz`, `/thumbnails/<ref>` serves the fixture thumb, `/ui` loads.

**(d) eval harness**
- **OR semantics:** multi-id `expected`, exactly one in top-5 → hit; zero → miss; verify never AND.
- Per-modality attribution: multi-modality expected credited to the hitting modality; miss counts against all listed.
- Ablation modes selectable; each produces all 3 breakdowns.
- `dataset.py` validator rejects labels referencing ids absent from the index (typo guard).

---

## 9. Open risks / notes for the write-up

- **API-first vs local-inference constraint profile.** This pipeline is API-first (Cohere
  embed + rerank) so there's no local GPU *inference* bottleneck at query time — a different
  profile than local-inference projects. The one local-GPU touchpoint is the ingest-time VLM
  captioner on the 3050; it's batch work, off the query path. Worth contrasting explicitly.
- **moondream2 on 4GB:** run in fp16, two passes per image (dense description + verbatim
  label transcription), concat into `content_text`, cache by image content hash.
- **`.gitignore`** the LanceDB data dir, `THUMBNAILS_DIR`, caption cache, and `.env`.
```