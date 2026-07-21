# Multimodal Search for Engineers

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-296%20passing-brightgreen)

A search system that unifies PDFs, diagrams, tables, and code into **one
searchable index**: two embedding spaces (Cohere Embed v4 for page/diagram
*images*, OpenAI text-embedding-3-small for table/code/caption *text*), one
LanceDB table, hybrid retrieval (two vector retrievers + full-text, fused via
RRF) with Cohere Rerank v3 on top. Built to answer a question a
single-modality search tool can't: "which of my papers, diagrams,
spreadsheets, and source files actually talks about this," in one query.

## Screenshots

The idle search page:

![Idle search page](docs/screenshots/01-idle-search.png)

A PDF result for "what is low-rank adaptation of large language models":

![PDF page result for a LoRA query](docs/screenshots/02-pdf-result.png)

A diagram result with its thumbnail, for "diagram showing the transformer
encoder decoder architecture":

![Diagram result with thumbnail for a transformer architecture query](docs/screenshots/03-diagram-result.png)

A code result with the mono snippet, for "infer implementation flow from
session text":

![Code result with a monospace snippet](docs/screenshots/04-code-result.png)

## Architecture at a glance

![retrieval pipeline](corpus/docs/retrieval_pipeline_diagram.png)

Query → embed with both providers (Cohere text query-embed, OpenAI text
query-embed) → two vector searches + full-text search (all three against the
same LanceDB table) → three-way Reciprocal Rank Fusion → Cohere Rerank v3 →
top-k results with thumbnails. PDFs and diagrams are embedded as *images*
(ColPali-style, single-vector, Cohere Embed v4) rather than OCR'd text;
diagrams and scanned pages also get a second OpenAI vector over their
VLM caption text. Tables are serialized to markdown and capped at 200 rows;
code is chunked by tree-sitter symbol boundaries (function/class), not
fixed-size splits -- both embedded via OpenAI text-embedding-3-small.
Text-less images (diagrams, scanned pages) get a moondream2-generated
caption so full-text search, the OpenAI vector, and reranking all have real
text to work with. Full design rationale is in [`PLAN.md`](PLAN.md) and
[`EMBEDDING_MIGRATION_PLAN.md`](EMBEDDING_MIGRATION_PLAN.md).

## Setup

```
pip install -e ".[dev]"
```

Build the frontend (requires Node 20+). The UI is a React + Vite app in
[`frontend/`](frontend); its build output is gitignored, so **a fresh clone
serves nothing at `/ui` until this runs**:

```
cd frontend && npm install && npm run build
```

That emits into `src/mmsearch/api/static/`, which FastAPI already serves: no
separate frontend server in production. Design rationale and the dev-server
workflow are in [`FRONTEND_PLAN.md`](FRONTEND_PLAN.md).

Create `.env` at the repo root (see [`.env.example`](.env.example)):

```
COHERE_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
MMSEARCH_API_KEY=your-own-shared-secret-here
```

`MMSEARCH_API_KEY` gates `/search` and `/thumbnails` -- generate one with:

```
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

The server won't start without it. Opening `/ui` in a browser prompts for the
key on first search and remembers it (`localStorage` + a cookie, so both
`fetch` calls and `<img>`-loaded thumbnails authenticate); calling `/search`
directly needs an `X-API-Key: <key>` header, e.g.:

```
curl -H "X-API-Key: your-own-shared-secret-here" "http://127.0.0.1:8000/search?q=auth"
```

Populate `corpus/` with your own PDFs (`specs/`), diagrams (`docs/`),
tables (`data/`), and code (`src/`). Three files are gitignored by license
or size rather than committed. See [`corpus/README.md`](corpus/README.md)
for exactly which ones, why, and where to source them.

## Running it

```
python -m mmsearch.ingest.cli ingest corpus/
uvicorn mmsearch.api.server:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/ui** and search. (If that page is blank or 404s,
the frontend build step above hasn't been run.)

To iterate on the UI itself, run Vite's dev server alongside uvicorn. It proxies
`/search` and `/thumbnails` to port 8000 and gives hot reload:

```
cd frontend && npm run dev      # http://127.0.0.1:5173
```

## Eval results

> **Pending re-measurement.** The table and "Known limitations" narrative
> below were measured against the old single-embedding-space (Cohere-only)
> index. The Cohere/OpenAI split changes the vector retrieval signal, so
> these numbers are stale until the corpus is re-ingested and re-evaluated
> (see [`EMBEDDING_MIGRATION_PLAN.md`](EMBEDDING_MIGRATION_PLAN.md) §8).

Hit-rate@5 against 25 hand-written labels
([`eval/labels.yaml`](src/mmsearch/eval/labels.yaml)) on the real ingested
corpus, across all three retrieval modes:

| | vector-only | rrf-only | rrf+rerank |
|---|---|---|---|
| **Aggregate** | 0.960 | 0.960 | 0.920 |
| *per-modality* | | | |
| - code | 1.000 | 1.000 | 1.000 |
| - diagram | 0.750 | 0.750 | 0.750 |
| - pdf_page | 0.857 | 0.857 | 0.714 |
| - table | 1.000 | 1.000 | 1.000 |
| *per-text_source* | | | |
| - code_source | 1.000 | 1.000 | 1.000 |
| - pdf_text_layer | 0.857 | 0.857 | 0.714 |
| - table_markdown | 1.000 | 1.000 | 1.000 |
| - vlm_caption | 0.750 | 0.750 | 0.750 |

(`rrf-only` and `rrf+rerank` measured at `RRF_K=20`, the current default.
See `config.py`'s comment for the controlled k=60-vs-k=20 comparison that
justified it. `vector-only` is unaffected by `RRF_K`.)

## Known limitations

**Sparse-text modalities lose fusion ties to dense-text competitors.**
RRF fuses by rank position across the vector and full-text lists. A
diagram's caption is a few sentences; a paper page is a few hundred words
of running text on the same topic. When both are relevant, the page tends
to rank consistently well in *both* retrievers while the diagram is
merely decent in one, and RRF systematically rewards consistency across
both signals over strength in one. This isn't a bug in the fusion math;
it's what rank-based fusion does when the two candidates' text volumes are
this asymmetric. Diagnosed directly: one query never surfaces a diagram
whose caption verbatim contains the query's own key phrase, because the
caption ranks only 14th of 50 in full-text search even though it's 7th of
50 in vector search: a real content match on the *word*, but a fused
ranking penalty for the *modality*.

**Cross-modal diagram search is sensitive to query phrasing.** The eval
label `"diagram showing the transformer encoder decoder architecture"`
correctly surfaces the right diagram at rank 2. The shorter, more casual
`"transformers architecture"` does not surface it in the top 5 at all:
same underlying image embedding, same index, different query wording. PDF
and code search are comparatively robust to phrasing; diagram retrieval
(image-embedding-to-text-query similarity) is the one path where how you
ask matters more than what you're asking about.

**Reranking can demote a correct top-ranked PDF page.** For the query
`"how to fine-tune an LLM for free using a Kaggle GPU"`, both `vector-only`
and `rrf-only` correctly rank the paper's actual intro page (`p1`) at rank
1, but `rrf+rerank` drops it out of the top 5 entirely, in favor of denser
mid-document pages that share more surface vocabulary with the query. This
is the one case on this eval set where reranking looks like a genuine
regression, not a small-sample fluke: the correct page and the promoted
pages are all topically relevant, but the reranker's judgment of "most
relevant" doesn't match the eval label's ground truth for a tutorial-style
document where the intro page is mostly setup rather than dense keyword
content.

## Tests

296 tests, all green, all run against fakes/fixtures. No real API calls,
no torch/GPU load except when actually exercising the local captioner:

```
pytest
```
