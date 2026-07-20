import type { Modality, TextSource } from "./modality";

/**
 * Mirrors the `SearchResult` dataclass in src/mmsearch/retrieve/types.py.
 * Field names and types must stay in step with it.
 */
export interface SearchResult {
  id: string;
  modality: Modality;
  score: number;
  snippet: string;
  thumbnail_ref: string;
  source_path: string;
  text_source: TextSource;
}

const API_KEY_STORAGE_KEY = "mm_api_key";

/**
 * Read the shared gate key. Persisted key is also mirrored into an
 * `mm_api_key` cookie by setApiKey -- <img src="/thumbnails/...">  can't
 * attach a custom header, only the browser's own cookie jar reaches that
 * request, so the backend accepts either.
 */
export function getApiKey(): string {
  return localStorage.getItem(API_KEY_STORAGE_KEY) ?? "";
}

export function setApiKey(key: string): void {
  localStorage.setItem(API_KEY_STORAGE_KEY, key);
  document.cookie = `mm_api_key=${encodeURIComponent(key)}; path=/; SameSite=Strict`;
}

/** Thrown for a 401 specifically, so callers can prompt for a key instead of
 * showing a generic "search failed" message. */
export class UnauthorizedError extends Error {
  constructor() {
    super("Invalid or missing API key.");
    this.name = "UnauthorizedError";
  }
}

/**
 * Relative URL on purpose: same-origin in production (FastAPI serves the built
 * bundle at /ui), and proxied to :8000 by the Vite dev server. One code path,
 * no environment switch.
 */
export async function search(
  query: string,
  signal?: AbortSignal,
): Promise<SearchResult[]> {
  const response = await fetch(`/search?q=${encodeURIComponent(query)}`, {
    headers: { "X-API-Key": getApiKey() },
    signal,
  });
  if (response.status === 401) {
    throw new UnauthorizedError();
  }
  if (!response.ok) {
    throw new Error(`Search failed (${response.status})`);
  }
  return (await response.json()) as SearchResult[];
}
