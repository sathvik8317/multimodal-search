# Frontend Plan

Design rationale for the search UI. The backend build is documented separately
in [`PLAN.md`](PLAN.md) — this document does not touch it.

## Context

The backend is complete and tested: FastAPI + Cohere Embed v4 + Rerank v3 +
LanceDB, hybrid RRF retrieval. `/search`, `/healthz`, and the `/thumbnails`
static mount are stable contracts and are **not modified** by this work. This is
a frontend-only upgrade.

The previous frontend was a single 120-line `index.html`. It worked — it proved
retrieval worked — but it was visually plain: no styling system, no component
structure, no loading or error states.

**Stated honestly:** that vanilla page was not technically incapable of looking
good. What it lacked was design, not architecture. React is chosen here largely
for portfolio signal, and because it keeps the polish work organized as
per-modality card variants and interaction states multiply. Both are real
reasons; neither is a claim that vanilla could not have gotten there.

## Decisions

### Framework: React + Vite + TypeScript

Angular's cost here is not bundle size, it is ceremony — DI, RxJS, NgModules,
CLI codegen. That machinery is designed to amortize across dozens of screens
maintained by a team. This application has one screen. Every Angular concept
this UI would touch is overhead with nothing to amortize against.

React with `useState` covers the entire state surface: a query string, a result
list, and a status enum. TypeScript comes free with the Vite template and keeps
the `SearchResult` shape honest against the backend dataclass.

### Styling: Tailwind

Rejected alternatives:

- **Component library (MUI, Chakra).** A heavy dependency whose output looks
  like default MUI. For a portfolio piece, looking like every other MUI app is
  the opposite of the goal.
- **Hand-rolled CSS.** Genuinely competitive at this size. Tailwind wins on
  three specifics that carry most of the visual polish: a consistent spacing
  scale, dark mode, and responsive breakpoints.

### Serving: both dev proxy and static build

Not an either/or choice — the two solve different problems.

**Production.** Vite builds into `src/mmsearch/api/static/`, which FastAPI
already serves. Zero backend changes; `create_app` in
[`src/mmsearch/api/main.py`](src/mmsearch/api/main.py) continues to mount that
directory at `/ui` exactly as before.

**Development.** `npm run dev` serves on `:5173` and proxies `/search` and
`/thumbnails` to the backend on `:8000`, giving HMR while iterating on design.
Two processes during development, one in production.

> **Gotcha:** the mount is at `/ui`, not `/`. `vite.config.ts` sets
> `base: '/ui/'`. Without it every built asset path 404s.

### Build artifacts: gitignored

Build output is **not** committed. The root `.gitignore` carries
`src/mmsearch/api/static/*` with a `!.gitkeep` negation, keeping the project's
ignore rules in one file.

The `.gitkeep` is load-bearing, not cosmetic: `StaticFiles(directory=...)` raises
at app-creation time if the directory does not exist, so without a tracked file
holding the directory open, **every** test fails on a fresh clone — not just the
frontend ones. Verified by deleting the build output and running the suite: 239
passed with only `.gitkeep` in place.

`emptyOutDir: true` is required (stale hashed assets would otherwise accumulate
forever) and it deletes everything in the directory except `.git` — including
`.gitkeep`. Vite copies `frontend/public/` into `outDir` *after* emptying, so
`.gitkeep` lives there too and is re-emitted on every build. Verified, not
assumed: the first implementation attempt asserted Vite spares dotfiles, and it
does not.

Consequence, documented in the README: a fresh clone must run `npm install &&
npm run build` before `/ui` serves anything.

---

## Design direction

The visual language is a dark-first developer-tool aesthetic, rendered in both
light and dark. Decisions below are fixed; they are not to be re-litigated
during implementation.

### Palette

Base palette, expressed as CSS custom properties in `index.css` and consumed
through Tailwind theme extensions rather than raw hex in components:

| Token | Dark | Light | Role |
|---|---|---|---|
| `--color-background` | `#0F172A` | `#F8FAFC` | Page background |
| `--color-foreground` | `#F8FAFC` | `#0F172A` | Primary text |
| `--color-accent` | `#22C55E` | `#15803D` | Focus rings, submit button, active state |
| `--color-muted` | `#1E293B` | `#E2E8F0` | Card background, skeletons |
| `--color-border` | `#334155` | `#CBD5E1` | Card and input borders |

Light mode is not an afterthought: a portfolio piece is opened by strangers on
unknown displays in unknown lighting. Both modes are driven by
`prefers-color-scheme`. The five base tokens swap through plain CSS variables so
a single `bg-bg` / `text-fg` utility is correct in both themes; Tailwind's
`dark:` variant is used where a token swap isn't enough, notably the four badge
hues.

Note the light accent is `#15803D` (green-700), not the `#22C55E` base: white
text on `#22C55E` measures 3.4:1, below the 4.5:1 requirement. Dark mode keeps
`#22C55E`, paired with a near-black green foreground at 8.8:1.

The accent green is reserved for interactive affordances — focus, submit, active
state. It is **not** used for modality badges (see below), so that "green" always
means "you can interact with this."

### Typography — split by modality

A single typeface for everything would be wrong here. What a result card shows
depends on its modality, and the two categories want different faces:

| Content | Face | Applies to |
|---|---|---|
| Prose snippets | Sans (Inter) | `pdf_page`, `diagram` |
| Literal-text snippets | Mono (JetBrains Mono) | `table`, `code` |
| `source_path`, `id` | Mono (JetBrains Mono) | **all** modalities |

Reasoning: `pdf_page` and `diagram` snippets are running prose — a paper's text
layer, or a VLM-generated caption. Mono at paragraph length measurably hurts
readability, so those get a proper sans. `table` snippets are serialized markdown
and `code` snippets are source text; both are column- and indentation-sensitive,
so they need mono to read correctly. Paths and ids are literal identifiers
regardless of what modality produced them, so they are always mono.

This routing is a lookup keyed on `Modality`, not a chain of conditionals.

### Modality badges — one color each

Four modalities, four distinct hues, so a scanning eye can tell result types
apart without reading the label:

| Modality | Hue |
|---|---|
| `pdf_page` | blue |
| `diagram` | purple |
| `table` | amber |
| `code` | emerald |

Distinct-per-type rather than one accent color everywhere: the whole premise of
this project is unifying four modalities in one result list, so the UI should
make the modality mix legible at a glance. Emerald for `code` sits in the green
family but is deliberately offset from the `#22C55E` interaction accent.

Badge colors come from a single lookup object keyed on `Modality`. Each pairing
meets 4.5:1 against its badge background in both themes.

### Quality checklist

Every item is a build requirement, verified before this work is called done:

- [ ] Visible focus states on all interactive elements — focus rings are never removed
- [ ] `prefers-reduced-motion` respected on every transition
- [ ] Text contrast ≥ 4.5:1 in **both** light and dark
- [ ] Responsive at 375px, 768px, 1024px, 1440px
- [ ] SVG icons (inline, Lucide-derived paths) — never emoji as icons
- [ ] Hover transitions 150–300ms

---

## Repo structure

```
frontend/                    # npm lives here, outside the Python package
  index.html                 # inline SVG data-URI favicon (no favicon file)
  package.json
  vite.config.ts             # base:'/ui/', outDir -> static dir, dev proxy
  tsconfig.json
  public/
    .gitkeep                 # re-emitted into outDir after emptyOutDir wipes it
  src/
    main.tsx
    App.tsx                  # useState only: query, results, status
    index.css                # @import "tailwindcss" + @theme palette tokens
    api.ts                   # search() + SearchResult type
    modality.ts              # badge colors + typography routing, keyed on Modality
    components/
      SearchBar.tsx
      ResultsList.tsx        # owns loading / empty / error states
      ResultCard.tsx
      ModalityBadge.tsx

src/mmsearch/api/static/     # Vite build target
  .gitkeep                   # tracked — holds the directory open
```

No `tailwind.config.js` and no PostCSS config: Tailwind v4 is CSS-first, so
theme tokens are declared in `index.css` via `@theme` and the build is wired
through `@tailwindcss/vite`.

## Component breakdown

| Component | Responsibility |
|---|---|
| `App.tsx` | Three `useState`s: `query`, `results`, `status` (`idle`/`loading`/`error`/`done`). Submit handler calls `api.ts`. No other logic. |
| `api.ts` | One `search(q)` function plus the `SearchResult` type. Throws on non-OK so `App` can set `error`. |
| `modality.ts` | Single source of truth for per-modality badge hue and snippet font. Both lookups keyed on the `Modality` string union. |
| `SearchBar.tsx` | Controlled input + submit. Autofocus, visible focus ring. |
| `ResultsList.tsx` | Switches on `status`: skeletons while loading, empty-state message, error message, or the mapped list. |
| `ResultCard.tsx` | Thumbnail, snippet, source path, score, modality badge. Graceful fallback when `thumbnail_ref` is absent. |
| `ModalityBadge.tsx` | Colored pill, hue from `modality.ts`. |

`SearchResult` mirrors the backend dataclass in
[`src/mmsearch/retrieve/types.py`](src/mmsearch/retrieve/types.py) exactly:
`id`, `modality`, `score`, `snippet`, `thumbnail_ref`, `source_path`,
`text_source`. `Modality` is a `str` enum backend-side, so these arrive as the
literals `"pdf_page" | "diagram" | "table" | "code"`.

### Thumbnail URLs must be percent-encoded

`make_id()` in `schema.py` builds pdf_page refs containing `#` — e.g.
`specs/paper.pdf#p5.png`, and the thumbnail files on disk are named that way
too. Interpolated raw into a URL, `#` starts a fragment: the browser requests
`/thumbnails/specs/paper.pdf` and silently drops `#p5.png`. Confirmed at the
HTTP level — unencoded 404s, `%23`-encoded returns 200.

**This bug predates the rewrite.** The previous vanilla `index.html` built
`img.src` by raw interpolation, so every PDF-page thumbnail was broken there
too; it showed a broken-image glyph rather than an obvious failure.
`ResultCard.tsx` encodes per path segment, preserving `/` while escaping `#`.
Anything else that builds a thumbnail URL must do the same.

## Scope

**In:** one search page — query box, result cards with thumbnails, modality
badges, score display, loading skeleton, empty state, error state, responsive
layout.

**Out, deliberately:** router (one route), state management library, react-query
(one fetch), pagination, auth, any new backend endpoint, multi-route navigation.

## Test change

`test_ui_returns_html_with_search_input_and_fetch_call` in
`tests/unit/test_api_main.py` asserted that the `/ui` body contained `<input`,
`/search`, and `fetch(`. A Vite build emits a bundled `/assets/index-<hash>.js`,
so `fetch(` never appears in `index.html` — this test fails regardless of the
build-artifact decision. It was asserting on a hand-written artifact that stops
existing the moment `/ui` serves a bundle.

**Removed.** There is nothing project-specific left to assert about one line of
framework mount configuration. The `/thumbnails` mount test remains — that one
serves real application data.

Suite: 240 → 239.

## Verification

1. `cd frontend && npm install && npm run build` — output lands in
   `src/mmsearch/api/static/`, and the built `index.html` references
   `/ui/assets/...` (not `/assets/...`).
2. `uvicorn mmsearch.api.server:app` → `http://127.0.0.1:8000/ui` loads CSS and
   JS with no 404s.
3. A real query renders thumbnails, correct badge hues, and scores; prose
   snippets render sans, table/code snippets render mono.
4. `npm run dev` on `:5173` with the backend on `:8000` — proxy returns results,
   HMR works.
5. All four states exercised: loading, results, empty (nonsense query), error
   (backend stopped mid-query).
6. Layout holds at 375px, 768px, 1024px, 1440px.
7. Design checklist above fully ticked.
8. `pytest` → 239 passed.
9. `git status` after a build shows nothing under `src/mmsearch/api/static/`
   except the tracked `.gitkeep`.
