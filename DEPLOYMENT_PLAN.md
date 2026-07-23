# Deploy multimodal-search to Render

## Context

The project runs locally today: a FastAPI service (`uvicorn mmsearch.api.server:app`)
serving `/search` over a LanceDB index built by a local ingest run, with the Vite/React
bundle mounted at `/ui`. The goal is a publicly-reachable deployment on Render that
serves the **already-built** index — 113 rows with real Cohere + OpenAI spend already
sunk into them — without re-running ingest on Render's infrastructure. 76 of those 113
rows end up committed (see §2): the other 37 are page text from two arXiv PDFs
`corpus/README.md` already excludes from redistribution, and that exclusion has to
reach the LanceDB rows derived from them, not just the source files and thumbnails.

Two premises in the original framing turned out not to survive contact with the code,
and the plan below is built on what's actually there instead:

- **The persistent disk is not load-bearing.** `data/` is 9.8 MB and is *read-only at
  query time* (`retrieve/pipeline.py` contains no write path; `db.open_table()` only
  reads; `create_app` only `mkdir`s the thumbnails dir). It's a build artifact, not
  mutable state. Render's disks also can't be written during the build at all, so a
  disk would have to be populated post-deploy over SSH — which requires a paid
  instance. Decision: **no disk**, data ships in the repo.
- **The lean production install already exists.** `torch`/`transformers` are not in
  `[project.dependencies]` — they're the `[local-captioner]` extra plus a manual torch
  install, and `clients/captioner_local.py:102` imports them lazily. Nothing to strip.

## Decisions locked in brainstorming

| Decision | Choice |
|---|---|
| Instance tier | **Free** (512 MB / 0.1 CPU, $0/mo, spins down after 15 min idle) |
| Data delivery | **Commit `data/lancedb` + `data/thumbnails` to git**; no persistent disk |
| License-excluded thumbnails | **Omit** the 37 LoRA + Attention page rasters; UI falls back to placeholder |
| License-excluded LanceDB rows | **Rebuild `data/lancedb` from scratch** to exclude the same 37 rows' text (113 → 76); `delete()` verified unsafe for this, see §2 |
| Frontend build | **On Render**, in the Python runtime's build command (`node`/`npm` are present) |
| Production deps | `pip install -e .` only — no `[local-captioner]`, no torch |

---

## The six questions, answered

### 1. Production dependencies — does the deployed service need `[local-captioner]`?

**No. And no change to `pyproject.toml` is needed, because the split already exists.**

`torch` isn't even a declared dependency — `pyproject.toml:37-40` puts only
`transformers` and `einops` in the `local-captioner` extra, with a comment directing
you to install torch separately. `clients/captioner_local.py:102-103` imports
`torch`/`transformers` *inside* `_ensure_model_loaded`, and `ingest/cli.py` imports the
real clients only inside `_run_ingest_command`'s production branch.

Verified at runtime rather than inferred — loading the production entrypoint and
inspecting `sys.modules`:

```
$ python -c "import mmsearch.api.server; ..."
torch loaded? False   transformers? False
pymupdf loaded? False  tree_sitter? False
import + table-open: 2.81s
resident memory:     134 MB
```

So `pip install -e .` is already the lean install. 134 MB fits the Free tier's 512 MB
cap with room to spare.

**The tradeoff you asked about — supporting live re-ingestion on the server:** don't.
It isn't a close call on any Render tier you'd want to pay for:

- The CPU-only torch wheel is ~200 MB installed, blowing up build time and image size
  for a code path that would never run at query time.
- moondream2's weights (~3.7 GB) download from HuggingFace **at first use**, into a
  cache directory that is ephemeral without a disk — so they'd re-download on every
  deploy, every spin-up.
- Peak captioning memory is multiples of the 512 MB Free cap. You'd need Standard
  (2 GB, $25/mo) at minimum, and probably more.
- Captioning on 0.5 shared vCPU is minutes per image. The corpus has 82 images.

Ingest stays local, where the GPU is. The deployed service is a read-only query surface.
`pymupdf` and `tree-sitter-language-pack` are also ingest-only but remain in base
`dependencies`; leave them — they cost build seconds, not runtime memory (confirmed
unloaded above), and splitting them out is churn for no measured gain.

### 2. Getting the existing index onto Render

**Mechanism: commit it. Render clones the repo at build time; the data is simply there.**

The alternative you had in mind — persistent disk — is ruled out by Render's own docs:

> "The disk cannot be accessed during build or pre-deploy commands, which run
> separately." — [render.com/docs/disks](https://render.com/docs/disks)

So a disk can only be populated *after* a successful deploy, via SCP over Render SSH
(paid instances only) or magic-wormhole. It also disables zero-downtime deploys and
pins the service to a single instance. For 9.8 MB of never-mutated data, that's cost and
constraint bought for nothing.

Worth knowing for its own sake: if you *had* gone the disk route, an empty disk would
**not** crash the service. `db.open_table()` (`src/mmsearch/db.py:13-17`) creates an
empty table when the name is absent, so the app would boot healthy and silently return
zero results. Silent, not loud — a good reason to avoid the path.

**Portability was verified, not assumed.** All 113 rows store POSIX-relative paths:

```
id            'tbl:data/car_prediction_data.csv'   backslashes: 0  absolute: 0
source_path   'data/automobile_dataset.csv'        backslashes: 0  absolute: 0
thumbnail_ref 'specs/2106.09685v2.pdf#p1.png'      backslashes: 0  absolute: 0
indices       content_text_idx  FTS  113 indexed / 0 unindexed
```

The Windows-built index and its in-table FTS index move to Linux unchanged.

**The redistribution exclusion has to reach `data/lancedb`, not just the thumbnails.**
`corpus/README.md` gitignores `2106.09685v2.pdf` (LoRA) and
`NIPS-2017-attention-is-all-you-need-Paper.pdf` (Attention Is All You Need) because
arXiv's license grants arXiv hosting rights, not third-party redistribution rights.
Omitting their 37 page-raster thumbnails (26 + 11) addressed the *image* form of that
content. It did not address the *text* form: `data/lancedb`'s `content_text` column
stores each page's extracted text layer verbatim (`ingest/documents.py:39`,
`content_text = page.get_text()`), and those 37 rows are still in the table headed for
commit. Same principle `corpus/README.md` already applies to the PDFs themselves —
"don't redistribute this" doesn't stop being true because the redistribution is a
LanceDB row instead of a raster or the original file.

Verified the exact scope directly against the table rather than assuming it matches the
thumbnail count:

```
total rows: 113
rows from excluded PDFs: 37   (2106.09685v2.pdf: 26, NIPS-2017-...-Paper.pdf: 11)
remaining after exclusion: 76
```

It matches 1:1 because pdf_page ingestion is one row per page, same as the thumbnails.

**`table.delete()` looked like the obvious tool here and turned out to be unsafe for
this specific requirement — verified, not assumed.** LanceDB's delete writes a deletion
vector; it does not remove the row's bytes from the underlying fragment file or from
older version manifests. Confirmed empirically on a throwaway table: after
`t.delete("text = 'SECRET_MARKER_ALPHA'")`, `count_rows()` correctly dropped, but a raw
byte-scan of the table's own directory still turned up the deleted string in three
places:

```
scratch_lance_test/t.lance/data/<fragment>.lance          <- original data file, untouched
scratch_lance_test/t.lance/_transactions/1-....txn        <- the delete transaction
scratch_lance_test/t.lance/_versions/....manifest         <- the pre-delete version
```

Committing a `delete()`-filtered table to git would put the excluded text right back
into the commit, just harder to notice. The fix is to never write it into this table's
history in the first place: filter the arrow data down to the retained rows, then
`create_table()` a **brand-new** table directory from that filtered data — a fresh
table has exactly one fragment containing only the rows it was given, no tombstones, no
prior versions. Verified against a real copy of the local table:

```
original rows: 113   filtered rows: 76
new table rows: 76
new table indices: content_text_idx FTS, 76/76 indexed
byte-level scan of the new directory for either excluded filename: NONE
```

**`scripts/rebuild_lancedb_for_commit.py`.** Reads `db.open_table()`, filters out rows
whose `source_path` is one of the two excluded PDFs, writes a fresh table to a temp
directory via `create_table(..., data=filtered)`, rebuilds the FTS index with the
existing `db.ensure_fts_index()`, then atomically swaps it in for `data/lancedb`.
Re-running it is idempotent (no matching rows on a second run is a no-op, not an error).

This is not a one-time script — re-ingesting `corpus/` in the future (the developer's
local `corpus/specs/` legitimately contains both PDFs, per `corpus/README.md`'s own
reproduction instructions) regenerates all 113 rows. **Run this script before every
future commit of `data/lancedb`, not just this one.** Documented as a step in the
re-ingest workflow below, not automated into a git hook — one documented manual step is
proportionate here, a pre-commit hook policing a two-file exclusion list is not.

**The `.gitignore` change has a trap in it.** Git does not descend into an excluded
directory, so `/data/` plus `!/data/lancedb/` silently does nothing. The parent glob has
to be widened first:

```gitignore
# LanceDB index + thumbnails are committed: they are the deployed read-only
# artifact (see DEPLOYMENT_PLAN.md), rebuilt only by a local `mmsearch ingest`
# followed by scripts/rebuild_lancedb_for_commit.py. caption_cache stays out --
# ingest-time only.
/data/*
!/data/lancedb/
!/data/thumbnails/
# Page rasters of the two PDFs corpus/README.md excludes for redistribution
# reasons. Omitting them keeps that position consistent; ResultCard.tsx falls
# back to a placeholder icon via its existing onError handler. The matching
# LanceDB rows are excluded by scripts/rebuild_lancedb_for_commit.py, not by
# gitignore -- they live inside chunks.lance's data files, not as loose files.
/data/thumbnails/specs/2106.09685v2.pdf#*
/data/thumbnails/specs/NIPS-2017-attention-is-all-you-need-Paper.pdf#*
```

`/data/*` matches one path segment only, so nothing under the re-included directories is
caught by it. `#` mid-pattern is literal (only a leading `#` is a comment).

Net: 45 of 82 thumbnails committed (~4.5 MB) and 76 of 113 rows committed in
`data/lancedb`. Repo pack goes from 80 KiB to roughly 6 MB.

**Re-ingest workflow after this change:** run ingest locally as today, then
`python scripts/rebuild_lancedb_for_commit.py` **before** staging — skipping it silently
re-commits all 113 rows, including the 37 excluded ones, since a fresh ingest
regenerates them from the developer's local `corpus/`. Then
`git add data/ && git commit && git push`. Render redeploys with the new index. No API
spend on Render's side, ever.

### 3. Frontend build

**Works as a single chained build command — Node is available in the Python runtime.**

Render's native-runtimes doc lists the build toolset as including
`bun, node, npm, pnpm, typescript, webpack, yarn` — *regardless of selected runtime*.
No Docker needed.

```
Build command:  cd frontend && npm ci && npm run build && cd .. && pip install -e .
Start command:  uvicorn mmsearch.api.server:app --host 0.0.0.0 --port $PORT
```

`npm run build` (`tsc -b && vite build`) writes into `../src/mmsearch/api/static` per
`vite.config.ts:19`, which is exactly where `api/main.py:18` mounts `/ui` from. The
gitignored build output is regenerated on Render every deploy — no committed bundle.

Two version pins matter here, and one of them is a hard build-breaker:

- **`.python-version` must contain `3.12`.** Render's default is now **3.14.3** for
  services created on/after 2026-02-11. `pyarrow==17.0.0` publishes wheels only through
  `cp312` (confirmed against the PyPI JSON: `cp38`–`cp312`, `manylinux_2_17` and
  `manylinux_2_28`). On 3.14 pip would fall through to compiling Arrow from source and
  the build fails. `3.12` also matches local (3.12.8) exactly. Using `.python-version`
  rather than `PYTHON_VERSION` lets Render track the latest 3.12 patch without a pin to
  maintain.
- **Node needs no pin.** Render's default is 24.14.1 (services on/after 2026-04-21);
  local is 24.16.0; Vite 8's floor is 20.19. Already satisfied.

**`pip install -e .` is mandatory, not stylistic.** `config.py:3` computes
`PROJECT_ROOT = Path(__file__).resolve().parents[2]`, and every data path hangs off it.
Editable-installed, `__file__` stays at `/opt/render/project/src/src/mmsearch/config.py`
→ `PROJECT_ROOT` = `/opt/render/project/src` → `data/lancedb` resolves correctly. A
plain `pip install .` copies the package into `site-packages`, `PROJECT_ROOT` becomes a
directory inside the virtualenv, and LanceDB creates a fresh empty table there. The
service boots healthy and returns zero results for every query, with no error anywhere.

### 4. Environment variables

Set in the Render dashboard (Environment → Environment Variables):

| Variable | Value | Notes |
|---|---|---|
| `COHERE_API_KEY` | from local `.env` | image embed + rerank |
| `OPENAI_API_KEY` | from local `.env` | text embed |
| `MMSEARCH_API_KEY` | **generate a new one** | don't reuse the local-only secret |
| `PYTHON_VERSION` | *(omit)* | use the `.python-version` file instead |
| `MMSEARCH_ENV` | **leave unset** | see below |
| `MMSEARCH_ALLOWED_ORIGINS` | **leave unset** | see below |

`settings.py:23` sets `env_file=".env"`; a missing `.env` on Render is not an error —
pydantic-settings falls through to real environment variables.

**`MMSEARCH_ENV` should stay unset, despite the names.** Setting it to `prod` looks
correct and is actively worse. `get_settings()` (`settings.py:57-61`) raises a clear
startup error on a missing `MMSEARCH_API_KEY` **only when `env == "dev"`**. Under
`prod`, that guard is skipped, and `require_api_key` (`deps.py:22`) then denies every
request because `not settings.api_key` is true. A typo'd secret would produce a
service that deploys green and 401s everything, instead of a deploy that fails loudly.
Leave it at the `dev` default until `staging`/`prod` actually have behavior attached.

**`MMSEARCH_ALLOWED_ORIGINS` does not need the Render URL.** In production the UI and
the API are the *same origin* — `api/main.py` mounts the built bundle at `/ui`, and
`frontend/src/api.ts` fetches the relative path `/search`. Same-origin requests never
consult CORS. This is already stated in `SECURITY_PLAN.md` item 3 ("Honest assessment:
this is purely defensive. It fixes no active bug"), and deploying doesn't change it.
The localhost:5173 default stays useful for local dev and is harmless in production.
Adding the Render URL would be a no-op, not a fix.

### 5. Cost and tier

Render's `/pricing` page is client-rendered and doesn't fetch; these come from Render's
own docs and their own July 2026 article. **Confirm the instance number in the dashboard
before committing** — it's the one figure I'd want a second look at.

| Item | Current price | Source |
|---|---|---|
| Free instance | **$0** — 512 MB / 0.1 CPU, spins down after 15 min idle, 750 instance-hrs/mo per workspace | render.com/docs/service-types, /docs/free |
| Starter | **$7/mo** — 512 MB / 0.5 CPU, always on | corroborated by Render's own "$7 Starter + $6 Basic-256mb Postgres = $13/mo" example |
| Standard | **$25/mo** — 2 GB / 1 CPU | render.com/docs/service-types |
| Persistent disk | **$0.25 / GB / mo** | render.com article, Jul 2026 |
| Hobby workspace | **$0** — 5 GB bandwidth + 500 build-pipeline min included | same |
| Bandwidth overage | **$0.15 / GB** | same |
| Build minutes overage | **$5 / 1,000 min** | same |

**This deployment: $0/mo.** Free instance, no disk, Hobby workspace.

Headroom against the included Hobby allowances:

- **Bandwidth** — ~0.5 MB per search (5 thumbnails at ~105 KB, bundle cached after
  first load). 5 GB ≈ **10,000 searches/month**. Not a constraint.
- **Build minutes** — `npm ci` + vite build + pip install of lancedb/pyarrow/pymupdf/
  tree-sitter is roughly 4–6 min. 500 min ≈ **~85–125 deploys/month**.

**What Free actually costs you:** the 15-minute spin-down. A cold request pays container
start plus the 2.8 s import measured on this machine, scaled onto 0.1 CPU — realistically
**30–60 s for the first hit** after idle. Fine for a link you send someone with a
warning; poor for anything you'd demo live. Starter at $7/mo is the fix and requires
zero plan changes — it's a dropdown, since nothing here depends on a disk. Measure the
real cold start after the first deploy and decide then.

### 6. Security posture — what changes when this is genuinely public

`SECURITY_PLAN.md` opens with "This is a local-only, single-user demo. It is not deployed
and this plan does not assume it ever will be." That premise is now false. Reviewing
each of its four items against a public origin:

**Holds up better than expected:**

- **The API-key gate is the right shape and is now doing more work than it was.** It was
  designed against "a malicious page you visit while the server runs"; it now also
  handles the whole internet scanning port 443. `hmac.compare_digest` against a
  256-bit `token_urlsafe(32)` secret is not brute-forceable, and `deps.py` correctly
  fails closed when `api_key` is unset.
- **Rejected requests cost nothing.** `dependencies=[Depends(require_api_key),
  Depends(rate_limit)]` (`api/main.py`) resolves in order, so a 401 short-circuits
  before `rate_limit` and long before `search_fn`. Unauthenticated scanners never bill
  Cohere or OpenAI, and — usefully — never consume the rate-limit budget either, so
  they can't lock you out of your own service.
- **TLS is solved.** Render terminates HTTPS on `*.onrender.com` automatically. This was
  the item `SECURITY_PLAN.md` explicitly deferred as out of scope.
- **The `deps.py` `ponytail:` caveat is satisfied.** The in-memory rate limiter "does
  not hold across multiple uvicorn workers" — the start command runs a single worker,
  and with no disk we could scale horizontally, but won't. Keep it single-instance;
  if that ever changes, the limiter is the thing that breaks first.

**What genuinely changes, and should be flagged:**

1. **The single shared secret has no per-recipient revocation.** Local-only, "one key,
   one user" was exact. Public, every person you share the link with holds the same
   secret, it's stored in their browser's `localStorage`, and it travels over the
   network. Rotation is all-or-nothing: change the env var, redeploy, re-issue to
   everyone. Acceptable for a demo you share deliberately; it is not an access-control
   system, and shouldn't be described as one.
2. **A leaked key is a metered spend hole, not just a nuisance.** The rate limit is
   global at 20 requests / 60 s, and each authenticated search costs 1 Cohere image-space
   embed + 1 OpenAI text embed + 1 Cohere rerank. Sustained abuse ceiling: **28,800
   searches/day**. Locally the cap protected against your own runaway loop; publicly it's
   the only thing between a leaked secret and a real bill. Consider lowering
   `MMSEARCH_RATE_LIMIT_MAX` for the deployed instance — it's already an env var, no
   code change — and set billing alerts on both provider dashboards.
3. **Failed-auth attempts are not rate limited at all.** Consequence of the ordering in
   the "holds up" list above: it's the right call for spend, but it means unbounded 401
   handling. Not exploitable against a 256-bit key, and each 401 is cheap — but on a
   0.1 CPU Free instance a sustained flood is a plausible availability nuisance. Render
   has no WAF on Free. Accepted, documented, not fixed.
4. **The gate cookie was missing `Secure`.** `frontend/src/api.ts` wrote
   `mm_api_key=...; path=/; SameSite=Strict` with no `Secure` flag. Fixed as part of this
   change (verified the addition doesn't break local dev — both `http://localhost` and
   `http://127.0.0.1` are treated as secure contexts by the browser, so the cookie still
   sets over plain HTTP in dev).
5. **`/healthz` and `/ui` are unauthenticated by design** (`SECURITY_PLAN.md` item 1:
   "`/ui` must load in order to prompt for the key"). Still correct. It does mean the
   deployment is publicly enumerable as a search service. That's fine and unavoidable.
6. **`source_path` is returned to clients.** Repo-relative only (`data/car_prediction_
   data.csv`) — verified across all 113 rows, no absolute paths, no machine paths. No
   leak.
7. **No request logging or alerting.** If someone does burn quota you'd learn from the
   provider's bill. Render's log stream exists but nothing is instrumented. Out of scope
   here; worth noting as the gap it is.

**Bottom line:** nothing in `SECURITY_PLAN.md` *breaks* on becoming public — the API key
does the job it was designed for and TLS arrives free. What changes is the blast radius
of a leak, and the fact that "one shared secret" now describes a group rather than a
person. `SECURITY_PLAN.md` carries a short "post-deployment" note recording that its
opening premise no longer holds, pointing back at this document.

---

## Implementation

### Files changed

| File | Change |
|---|---|
| `.gitignore` | `/data/` → `/data/*` + the two re-inclusions + two raster exclusions (exact block in §2) |
| `scripts/rebuild_lancedb_for_commit.py` | rebuilds `data/lancedb` from scratch excluding the 37 rows sourced from the two license-restricted PDFs (§2) |
| `.python-version` | one line: `3.12` |
| `frontend/src/api.ts` | added `Secure` to the cookie attributes |
| `src/mmsearch/api/main.py` | `GET /` → `RedirectResponse("/ui/")` |
| `.env.example` | documents `OPENAI_API_KEY` as required (previously only Cohere) and that deployment sources these from the Render dashboard |
| `README.md` | "Deployment" section pointing at this document |
| `SECURITY_PLAN.md` | post-deployment note that the local-only premise no longer holds |
| `data/lancedb`, `data/thumbnails` | committed for the first time (76 rows, 45 thumbnails) |

`pyproject.toml` is **not** modified — §1 explains why nothing needs splitting out.

### Commit sequence

1. `.gitignore` + `.python-version` + `api.ts` + root redirect.
2. `python scripts/rebuild_lancedb_for_commit.py` — replaces the local `data/lancedb`
   with the filtered 76-row rebuild. **This changes the local working index, not just
   a staging copy** — the two excluded papers stop being searchable locally too, same
   tradeoff `corpus/README.md` already accepts for the source PDFs. A full 113-row
   backup was kept locally (`data/lancedb_full_backup_113rows`, gitignored) before the
   rebuild ran, so the pre-exclusion index is recoverable without re-paying for ingest.
3. `git add data/` — **verify the staged content before committing, not after**:
   - `python -c "import lancedb; print(lancedb.connect('data/lancedb').open_table('chunks').count_rows())"`
     must print **76**.
   - `git add --dry-run data/` should list the lancedb tree plus 45 thumbnails.
   - `git add --dry-run data/ | grep -c 'NIPS-2017\|2106.09685'` must be **0**.
4. Commit and push to `master` (per existing convention, no `Co-Authored-By` trailer).

### Render dashboard setup

1. New → Web Service → connect `sathvik8317/multimodal-search`, branch `master`.
2. Runtime **Python 3**, Instance type **Free**, Region: nearest.
3. Build command: `cd frontend && npm ci && npm run build && cd .. && pip install -e .`
4. Start command: `uvicorn mmsearch.api.server:app --host 0.0.0.0 --port $PORT`
5. Health check path: `/healthz` (exists, ungated, returns `{"status": "ok"}`).
6. Environment variables per the §4 table.
7. No disk. Auto-deploy on push to `master` is fine.

## Verification

**Before deploying — prove the committed-data path locally.** This is the check that
actually matters, because it catches both "the data didn't get committed" and the
`pip install -e .` trap in one shot:

```bash
git clone https://github.com/sathvik8317/multimodal-search.git /tmp/deploycheck
cd /tmp/deploycheck
python -c "import lancedb; t=lancedb.connect('data/lancedb').open_table('chunks'); \
           print(t.count_rows(), t.list_indices())"
# MUST print 76 (not 113) and the content_text_idx FTS index, 76/76 indexed.
# 0 rows => data/ didn't commit. 113 rows => rebuild_lancedb_for_commit.py wasn't run
# before this commit.
ls data/thumbnails/specs | wc -l          # expect 42 (26 ColPali + 16 Kaggle)
ls data/thumbnails/specs | grep -c NIPS   # expect 0

# Confirm the exclusion reached the table content, not just the thumbnail files --
# this is the specific gap this revision closes. Byte-level, not just a column check,
# since delete()-style approaches can leave content recoverable in raw files (see §2).
python -c "
import os
needles = [b'2106.09685v2.pdf', b'NIPS-2017-attention-is-all-you-need-Paper.pdf']
hits = []
for root, _, files in os.walk('data/lancedb'):
    for f in files:
        with open(os.path.join(root, f), 'rb') as fh:
            data = fh.read()
            hits += [(root, f, n) for n in needles if n in data]
print('byte-level hits for excluded filenames in data/lancedb:', hits or 'NONE')
"
# MUST print NONE.

cd frontend && npm ci && npm run build && cd .. && pip install -e .
python -c "from mmsearch import config; print(config.LANCEDB_URI)"
# MUST print the clone's data/lancedb, NOT a path inside site-packages.
```

**Existing suite must stay green** — `pytest` from the repo root, before and after.
Nothing in this plan touches retrieval, so any change there is a regression.

**After deploying:**

1. `curl https://<service>.onrender.com/healthz` → `{"status":"ok"}`.
2. `curl "https://<service>.onrender.com/search?q=test"` → **401** (gate is live).
3. Same with `-H "X-API-Key: <the Render value>"` → **200 with results**. Zero results
   here means the data didn't ship or the install wasn't editable — check §3.
4. Two known-good queries from the eval sweep, confirming both retrievers work through
   the deployed path:
   - `"car pricing dataset with kilometers driven and fuel type"` → top result
     `tbl:data/car_prediction_data.csv` (OpenAI text vector)
   - `"function that deduplicates a list while preserving order"` → top result
     `code:src/extractor.py#dedupe_preserve_order` (OpenAI text vector)
5. Open `https://<service>.onrender.com/ui/` in a browser: key prompt appears, key is
   accepted, results render, and **thumbnail images load** — the cookie path, which no
   unit test covers (`SECURITY_PLAN.md` test plan step 3). Confirm a LoRA or Attention
   result shows the placeholder icon rather than a broken image.
6. Check the Network tab for the `/search` request: same-origin, no CORS preflight.
   That's the evidence for §4's claim that `allowed_origins` is irrelevant here.
7. Time a cold request after 15+ min idle. Record the real number. If it's intolerable,
   switch to Starter — dashboard dropdown, no plan changes.

## Deferred, deliberately

- **Persistent disk** — no mutable state to persist. Revisit only if server-side
  re-ingestion ever becomes a goal, which §1 argues against.
- **Server-side ingest / `[local-captioner]`** — see §1. Ingest stays local.
- **Per-IP rate limiting, slowapi + Redis** — the `ponytail:` note in `deps.py`
  already names this; single-instance means the global counter still holds.
- **The query-embed failure gap** — `EMBEDDING_MIGRATION_PLAN.md`'s existing "Known gap"
  (neither `embed_query()` has the reranker's graceful fallback, so a provider failure
  is an unhandled 500) is unchanged by deployment, but it becomes user-visible rather
  than developer-visible. Still not blocking; worth doing before this URL goes anywhere
  that matters.
- **Request logging / spend alerting** — §6 item 7.
