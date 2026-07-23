# Plan: Authenticated `/upload` for the live Render deployment

## Context

The site is deployed on Render **Free** ($0/mo), serving a **read-only** LanceDB
index (76 rows) and thumbnails **committed to git** — no persistent disk, no
torch/moondream2 in prod. We want friends to upload real files (PDF, image,
code, Excel) via an authenticated endpoint on the public site, landing in the
same searchable index. That breaks four assumptions of the current deploy:
persistence, captioning memory, the shared content model, and the fact that
every route today is a read-only GET. Decisions are locked (below); this plan
turns them into an execution spec.

**Locked decisions:** (1) Storage → **Cloudflare R2** (keep Render Free, ~$0/mo).
(2) Captioning → **OpenAI vision** on the server path (`OPENAI_API_KEY` already
in prod); local moondream2 unchanged for local ingestion. (3) Content model →
**shared index, tagged rows** (no per-user isolation; moderate by predicate
delete). (4) Code → **uploaded `.py` file** through the existing tree-sitter
path; no paste box.

**Branch & gate:** this work happens entirely on `upload-feature-r2`, branched
off `master` at `f937f36`. No merge or push to `master` without explicit
instruction. Implementation does not start until explicit go-ahead is given
in addition to this plan being written — per the original request.

## Key facts grounding the design (verified in code)

- Single-file ingest primitive already exists: `_ingest_one(category, path, root, embedders, captioner, thumbnails_dir)` + `classify_file()` in `src/mmsearch/ingest/base.py`; `db.upsert(table, rows)` in `src/mmsearch/db.py`. `/upload` reuses these — no directory walk.
- Captioner is a clean `Captioner` Protocol (`caption(image_bytes)->str`, `src/mmsearch/clients/protocols.py`) — hosted impl is a drop-in, injected only server-side.
- Auth reuse is trivial: `Depends(require_api_key)` (`src/mmsearch/api/deps.py`) already reads header-or-cookie, fails closed. No server session exists (cookie is set by frontend JS).
- Storage is the **only** `lancedb.connect()` call — `db.open_table()` (`db.py:13`). It hard-codes `config.LANCEDB_URI` (local path, not env-driven). That single choke point is what makes R2 tractable.
- Rate limit is one **global** `deque` shared by `/search` (`deps.py:33`). Uploads need their own bucket.
- Not installed / not wired yet: `python-multipart`, `openpyxl`, `boto3`; CORS is **GET-only** (`main.py:53`); `.xlsx` returns `None` from `classify_file` (deliberately skipped today).

## Work breakdown

### 1. Storage → Cloudflare R2

- **LanceDB table on R2.** Add `lancedb_uri` to `Settings` (`settings.py`, env `MMSEARCH_LANCEDB_URI`), default the current local path so local dev is unchanged. `db.open_table()` sources its default from settings instead of `config.LANCEDB_URI`. Pass R2 creds via `storage_options` (endpoint, key, secret, `region="auto"`) or standard `AWS_*` env vars that LanceDB's object_store reads. Add a `create_if_missing: bool` param to `open_table`; **server passes `False`** so a missing/misconfigured R2 index fails loud instead of silently serving an empty table (the DEPLOYMENT_PLAN §2 failure mode).
- **One table, consolidated on R2.** Curated (76) + uploaded rows live together — accepts the +50–200ms/query latency already agreed. Seed once: `scripts/seed_r2.py` uploads the committed 76-row index to R2 **only if the R2 table is absent/empty** (idempotent, run manually once). Committed `data/lancedb` becomes the seed/backup.
- **Thumbnails split by origin.** Curated thumbnails stay **local** (shipped in git, served as today — fast, no egress). Uploaded thumbnails go to **R2** under the `uploads/` prefix via `boto3`. `/thumbnails/{path}` branches: `uploads/…` → stream bytes from R2 (keeps the API-key gate); else the existing local `FileResponse` + `_resolve_thumbnail` containment check.
- **Reader freshness.** After an upsert, call `table.checkout_latest()` (or reopen) so the module-level reader handle opened at server import sees new rows.

### 2. Captioner → OpenAI vision (server path only)

- New `src/mmsearch/clients/captioner_api.py`: `class ApiCaptioner` implementing the `Captioner` Protocol. Calls `gpt-4o-mini` vision with a base64 data-URI image and a describe+transcribe prompt mirroring `_compose_caption` (so caption style/quality stays consistent with local moondream2). Reuse the `_call_with_retry` backoff pattern from `clients/openai.py`.
- Wire only into the upload endpoint: construct `ApiCaptioner()` alongside the existing server-level `Embedders(image=CohereClient, text=OpenAIClient)`. Local CLI keeps `LocalCaptioner`.

### 3. `.xlsx` support

- Add `openpyxl` dep. Route `.xlsx → "table"` in `classify_file` (drop it from the "skipped" docstring).
- In `ingest/tables.py::ingest_table`, branch on suffix: `.csv` → `csv.reader` (as now); `.xlsx` → openpyxl active sheet → same `columns` + `data_rows` shape, **stringifying every cell** (openpyxl yields typed/None values; `_rows_to_markdown` does `str.join` and will crash otherwise — map `None`→`""`). Then reuse `_select_embedded_rows` + `_rows_to_markdown` unchanged (keeps the `MAX_TABLE_ROWS`/`MAX_TABLE_EMBED_CHARS` caps).

### 4. `POST /upload` endpoint + security

- Add `python-multipart`. Add `"POST"` to CORS `allow_methods`.
- Route in `api/main.py`: `dependencies=[Depends(require_api_key), Depends(upload_rate_limit)]`. Accepts `UploadFile` + optional `uploader: str = Form(None)`.
- **Size cap** — `MAX_UPLOAD_BYTES` (10 MB). Reject early on `Content-Length`, and **enforce while streaming** the read (don't trust the header). Over cap → 413.
- **Magic-byte sniffing** — small dict of byte-prefix signatures for the closed allowlist (`%PDF-`; PNG/JPEG/GIF/BMP/WEBP magic; `.xlsx` = `PK\x03\x04`). Text types (`.py`, `.csv`) → sample-decode as UTF-8, reject on NUL/undecodable bytes. Extension/content mismatch → 415. No `python-magic`/libmagic (a handful of prefix checks covers ~7 types).
- **Separate rate limit** — refactor the sliding-window logic into `_make_rate_limiter(max, window, hits)` returning a dependency; `/search` keeps its deque, `/upload` gets its own, stricter one (`upload_rate_limit_max` default **5** / 60s, env-overridable). Independent budgets.
- **Flow:** write validated bytes to a temp root at `<tmp>/uploads/<uploader-or-anon>/<sanitized_name>` (so `path.relative_to(root)` yields `uploads/…` → `source_path`/`id` carry the prefix) → `classify_file` (None → 415) → `_ingest_one(..., captioner=ApiCaptioner(), thumbnails_dir=<tmp_thumbs>)` → stamp `metadata.uploader` + `metadata.uploaded_at` on each row → `db.upsert` → boto3-PUT any files produced under `<tmp_thumbs>` to R2 at their relpath (thumbnail_ref already matches) → `checkout_latest()` → clean temp dirs.
- **Response** (drives the status indicator): `{status, filename, modality, rows_written}`; structured error body on 4xx.

### 5. Frontend

- `frontend/src/api.ts`: `upload(file, uploader?, signal)` → `POST /upload` with `FormData` + `X-API-Key` header. Map 401→`UnauthorizedError`, and surface 413/415/429 messages.
- New `frontend/src/components/UploadPanel.tsx`: file picker (accept the allowlist), optional uploader-name field, **modality-aware status indicator** (uploading → captioning/embedding → "Added N chunks as {PDF|image|code|table}", or the specific error). Match existing Tailwind styling.
- Render from `App.tsx` alongside `<SearchBar>`, behind the same key gate. Build still outputs to `src/mmsearch/api/static`.

### 6. Deps / config / deploy

- `pyproject.toml` base deps += `python-multipart`, `openpyxl`, `boto3`.
- Render env additions: R2 creds (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, `AWS_REGION=auto`, bucket), `MMSEARCH_LANCEDB_URI=s3://<bucket>/lancedb`, optional `MMSEARCH_UPLOAD_RATE_LIMIT_MAX`. `OPENAI_API_KEY` already present (now also captions).
- Run `scripts/seed_r2.py` once after setting env, before first upload.
- Verify the object-store stack works under the pinned `pyarrow==17.0.0` / Python 3.12.

## Critical files

- `src/mmsearch/settings.py`, `src/mmsearch/config.py`, `src/mmsearch/db.py` — R2 wiring, fail-loud open.
- `src/mmsearch/api/main.py`, `src/mmsearch/api/deps.py` — `/upload` route, second rate limiter, CORS POST, `/thumbnails` R2 branch.
- `src/mmsearch/ingest/base.py`, `src/mmsearch/ingest/tables.py` — `.xlsx` route + reader.
- `src/mmsearch/clients/captioner_api.py` (new) — OpenAI-vision `Captioner`.
- `src/mmsearch/storage/r2.py` (new, small) — boto3 thumbnail put/get, used by `/upload` and `/thumbnails`.
- `scripts/seed_r2.py` (new), `scripts/delete_upload.py` (new, one-line predicate delete for moderation).
- `frontend/src/api.ts`, `frontend/src/components/UploadPanel.tsx` (new), `frontend/src/App.tsx`.
- `pyproject.toml`, `DEPLOYMENT_PLAN.md` (env table + seed step), `SECURITY_PLAN.md` (upload attack surface).

## Verification

- **Unit tests:** `.xlsx`→markdown parity with the equivalent CSV; magic-byte validator (accept real PDF/PNG, reject a renamed `.exe`, reject a text file with NUL); `upload_rate_limit` trips at its cap independently of `/search`'s; `ApiCaptioner` returns non-empty against a mocked OpenAI response.
- **Integration:** `POST /upload` for each modality (small PDF, PNG, `.py`, `.csv`, `.xlsx`) against a temp **local** LanceDB uri + `FakeCaptioner` + fake/stubbed R2 → rows become searchable; `source_path` carries `uploads/`; `metadata` has `uploader`+`uploaded_at`. Oversized → 413; renamed binary → 415; over-cap burst → 429.
- **Existing suite** (301 tests) stays green; frontend `tsc -b && vite build` clean.
- **Live smoke (post-deploy + seed):** upload a file with a friend's key → searchable within one query; uploaded image thumbnail loads from R2, code/table falls back to the icon; curated-corpus search still returns its rows; confirm size cap, type rejection, and the stricter upload limit all fire against the live URL.

## Deliberate simplifications (ceilings)

- **One shared index, tagged rows** — no per-user isolation. Ceiling: needs real per-user auth + a moderation UI if it grows past trusted friends. Moderation today = `scripts/delete_upload.py` predicate delete.
- **LanceDB-on-object-store commits** — single-writer friends scale; concurrent uploads could clash on the R2 manifest commit, resolved by retry. Ceiling: a commit coordinator if write concurrency ever rises.
- **Curated search now pays R2 latency** — one consolidated table over two-tables-and-merge, trading ~100ms for a simpler hot path.
