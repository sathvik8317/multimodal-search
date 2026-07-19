import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// FastAPI mounts the built assets at /ui (see api/main.py), NOT at the site
// root. Without `base`, every emitted asset path is absolute-from-root and 404s
// in production while working fine in dev — set it and forget it.
const BASE = "/ui/";

// Build straight into the directory FastAPI already serves. Relative to this
// project root, which is what Vite resolves `outDir` against -- avoids needing
// @types/node just to compute one path.
//
// `emptyOutDir` is required (stale hashed assets would otherwise pile up
// forever) but it wipes EVERYTHING except .git -- including the tracked
// .gitkeep that holds the directory open for a fresh clone. public/ is copied
// into outDir after emptying, so keeping .gitkeep there re-emits it every
// build. Verified, not assumed: an earlier revision of this file claimed Vite
// spares dotfiles. It does not.
const OUT_DIR = "../src/mmsearch/api/static";

const BACKEND = "http://127.0.0.1:8000";

export default defineConfig({
  base: BASE,
  plugins: [react(), tailwindcss()],
  build: {
    outDir: OUT_DIR,
    emptyOutDir: true,
  },
  server: {
    // Dev server runs on :5173 and proxies data routes to uvicorn on :8000, so
    // relative fetches in api.ts work unchanged in both dev and production.
    proxy: {
      "/search": BACKEND,
      "/thumbnails": BACKEND,
      "/healthz": BACKEND,
    },
  },
});
