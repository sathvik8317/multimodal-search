# Security hardening pass — multimodal-search

## Context

This is a local-only, single-user demo. It is not deployed and this plan does not
assume it ever will be. The goal is to close four specific gaps at a scale
proportionate to that reality — not deployment readiness.

Out of scope, deliberately: HTTPS/TLS, multi-user auth, logging/observability
infrastructure, anything presuming a real deployment target.

Current state, verified in the code:

- `api/main.py:27` — `/search` is unauthenticated. Every call runs the real
  pipeline (`server.py:24` wires `mode="rrf+rerank"`), so each request costs
  1 Cohere embed + 1 Cohere rerank against a single API key.
- No rate limiting anywhere.
- No CORS configuration.
- No settings object at all. `config.py` holds only constants (paths, tuning
  knobs) and no secrets; `load_dotenv()` sits at import time in `server.py:15`
  and `ingest/cli.py:81`, and `clients/cohere.py:36` reads `os.environ` directly.

## Item 1 — Shared API-key gate

**Approach:** one secret in settings, checked by a FastAPI dependency using
`hmac.compare_digest`. No user accounts. This mirrors how the project already
trusts one secret for one resource (the Cohere key).

Gate covers `/search` and `/thumbnails`. `/healthz` and `/ui` stay open — `/ui`
must load in order to prompt for the key.

**The `<img>` constraint drove the design.** `ResultCard.tsx:55` renders
thumbnails as `<img src="/thumbnails/...">`, and browsers cannot attach custom
headers to `<img>` requests. A header-only gate would 401 every image. So the
dependency accepts the key from **either** `X-API-Key` **or** an `mm_api_key`
cookie; the frontend writes both when the key is saved. `SameSite=Strict` means
the cookie is never sent on cross-site requests, so this does not reopen the
cross-site quota-burn hole.

**Gating `/thumbnails` requires replacing the `StaticFiles` mount with a real
route** — `app.mount()` does not accept dependencies. `StaticFiles` was silently
providing path-traversal protection, so the replacement route must do it
explicitly: resolve the joined path and confirm it stays under `thumbnails_dir`
before returning a `FileResponse`. This is a trust boundary; it does not get
the lazy treatment.

## Item 2 — Rate limiting

**Approach:** stdlib `deque` of timestamps in a FastAPI dependency, applied to
`/search`. Roughly 12 lines, no new dependency.

**Reasoning over slowapi:** the cap that matters here is *global*, because it
protects Cohere quota and there is exactly one client. slowapi's per-IP keying
(`get_remote_address`) is close to meaningless in that setting, and it costs a
dependency plus `Limiter` setup, middleware, an exception handler, and a forced
`request: Request` parameter in gated signatures. The ceiling of the stdlib
version is real and gets a `ponytail:` comment naming it: in-memory and
per-process, so it does not hold across multiple uvicorn workers. Upgrade path
is slowapi + Redis if this ever runs multi-worker.

## Item 3 — CORS

**Honest assessment: this is purely defensive. It fixes no active bug.**

Neither current path is cross-origin. In production FastAPI serves `/ui` and
`/search` from one origin (`main.py:32-33`); in dev `vite.config.ts:34-38`
proxies `/search`, `/thumbnails`, `/healthz` to :8000, so the browser only ever
talks to :5173. Adding CORSMiddleware changes the behavior of zero current
requests.

It also does **not** fix the threat it superficially resembles. A malicious page
you visit while the server runs can issue a simple cross-origin GET to
`127.0.0.1:8000/search`; that request has no preflight, so it reaches FastAPI and
bills Cohere regardless of CORS. CORS only stops the attacker page from *reading*
the response. **The API key is what actually closes that hole**, by rejecting
before the pipeline runs.

What CORS buys is clarity: an explicit origin allowlist means that if someone
later runs the frontend without the proxy or on another port, they get a loud,
well-understood CORS error instead of a mystery. Six lines for that is worth it,
labeled as what it is.

**Approach:** `CORSMiddleware` with `allow_origins` read from settings,
defaulting to `["http://localhost:5173", "http://127.0.0.1:5173"]`.
`allow_credentials=True` (the cookie), methods limited to `GET`, headers to
`X-API-Key`. No wildcards.

## Item 4 — Typed settings

**Approach:** new `src/mmsearch/settings.py` with a `pydantic-settings`
`BaseSettings` model, exposed via an `@lru_cache`'d `get_settings()` rather than
a module-level instance — import-time instantiation would break every test that
lacks a key, and the cached accessor is overridable in tests via
`app.dependency_overrides`.

Fields: `cohere_api_key`, `api_key`, `allowed_origins`, `rate_limit_max`,
`rate_limit_window`, and `env: Literal["dev", "staging", "prod"] = "dev"`.

**The env field is set up, not built out** — as requested. Its only current
effect: in `dev` a missing `api_key` is a startup error with a clear message
telling you to set one; the `staging`/`prod` branches exist as declared values
with no behavior attached yet.

`config.py`'s constants stay put. They are not environment-driven, nothing reads
them from env today, and moving them would churn ~20 import sites and
`test_config.py` for no gain.

`python-dotenv` comes out of `pyproject.toml`; pydantic-settings reads `.env`
natively. `pydantic` is already present transitively via FastAPI.

## Files touched

**New**
- `src/mmsearch/settings.py` — `Settings`, `get_settings()`
- `src/mmsearch/api/deps.py` — `require_api_key`, `rate_limit`
- `tests/unit/test_settings.py`
- `.env.example`
- `SECURITY_PLAN.md` — this plan, written to the repo as requested
  (`PLAN.md` and `FRONTEND_PLAN.md` are not touched)

**Modified**
- `src/mmsearch/api/main.py` — CORS middleware; `dependencies=[...]` on
  `/search`; `/thumbnails` mount replaced with a gated `FileResponse` route
  including the containment check
- `src/mmsearch/api/server.py` — drop `load_dotenv`, read `get_settings()`
- `src/mmsearch/clients/cohere.py:36` — source the key from settings, keeping
  the existing `RuntimeError` message (`test_cohere_client.py:86` asserts on it)
- `src/mmsearch/ingest/cli.py:76-81` — drop the local `load_dotenv` import/call
- `pyproject.toml` — `+pydantic-settings`, `-python-dotenv`
- `frontend/src/api.ts` — attach `X-API-Key`; save key to localStorage **and**
  cookie; surface 401 distinctly from other failures
- `frontend/src/App.tsx` — small key input shown on 401, retry after save
- `frontend/src/components/ResultCard.tsx` — **unchanged** (that was the point
  of the cookie)
- `tests/unit/test_api_main.py` — existing tests need a key; add gate tests
- `tests/unit/test_ingest_cli.py:117-124` — rewrite the dotenv assertion
- `README.md` — document `MMSEARCH_API_KEY` setup

## Test plan

**Unit — `tests/unit/test_api_main.py`**
- `/search` without a key → 401; **assert the search fn was never called**
  (proves Cohere is not billed on a rejected request — the actual point)
- `/search` with a wrong key → 401
- `/search` with the correct key via header → 200
- `/thumbnails/x.png` with the key via **cookie only** → 200 (the `<img>` path)
- `/thumbnails/x.png` with no key → 401
- `/thumbnails/../../etc/passwd` and encoded variants → 404/403, never escapes
  `thumbnails_dir`
- `/healthz` with no key → 200 (stays open)
- Existing four tests updated to pass a key

**Unit — rate limit**
- `rate_limit_max + 1` calls in one window → last is 429
- Monkeypatch `time.monotonic` forward past the window → allowed again
- Deque does not grow unbounded across the window boundary

**Unit — `tests/unit/test_settings.py`**
- Reads from env; `.env` file honored via `tmp_path`
- Missing `api_key` in `env="dev"` → clear startup error
- `env` rejects a value outside the three literals
- `allowed_origins` parses to a list

**Unit — `tests/unit/test_cohere_client.py`**
- Existing missing-key `RuntimeError` test still passes against settings

**Manual end-to-end** (the parts tests cannot cover)
1. `uvicorn mmsearch.api.server:app` with `MMSEARCH_API_KEY` set
2. `curl localhost:8000/search?q=auth` → 401; `-H "X-API-Key: ..."` → results
3. Load `/ui`, confirm the key prompt appears, enter it, confirm results render
   **and thumbnail images load** (the cookie path — the failure mode this whole
   design exists to avoid)
4. `npm run dev` on :5173, confirm the proxied path behaves identically
5. Hammer search past the limit, confirm a 429 surfaces without a crash

## Verification

`pytest` (full suite, existing tests must stay green) plus the five manual steps
above. The manual thumbnail check in step 3 is not optional — no unit test
exercises a real browser `<img>` request.
