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
    signal,
  });
  if (!response.ok) {
    throw new Error(`Search failed (${response.status})`);
  }
  return (await response.json()) as SearchResult[];
}
