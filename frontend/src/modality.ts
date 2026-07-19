/**
 * Single source of truth for per-modality presentation.
 *
 * Mirrors `Modality` in src/mmsearch/schema.py. That enum subclasses `str`, so
 * these literals are exactly what arrives over JSON.
 */

export type Modality = "pdf_page" | "diagram" | "table" | "code";

export type TextSource =
  | "pdf_text_layer"
  | "vlm_caption"
  | "table_markdown"
  | "code_source";

export const MODALITY_LABEL: Record<Modality, string> = {
  pdf_page: "PDF page",
  diagram: "Diagram",
  table: "Table",
  code: "Code",
};

/**
 * One distinct hue per modality so the modality mix in a result list is legible
 * at a glance -- the whole premise of the project is unifying four sources in
 * one ranking, and a single accent color everywhere would hide that.
 *
 * Deliberately offset from the interaction accent (green): in this UI green
 * means "you can interact with this". `code` uses emerald, which reads as
 * distinct from the accent's #22C55E.
 *
 * Full literal class strings, not interpolated fragments -- Tailwind's scanner
 * only sees classes that appear verbatim in source.
 */
export const MODALITY_BADGE: Record<Modality, string> = {
  pdf_page:
    "bg-blue-100 text-blue-800 ring-blue-600/20 dark:bg-blue-400/15 dark:text-blue-300 dark:ring-blue-400/30",
  diagram:
    "bg-purple-100 text-purple-800 ring-purple-600/20 dark:bg-purple-400/15 dark:text-purple-300 dark:ring-purple-400/30",
  table:
    "bg-amber-100 text-amber-900 ring-amber-600/20 dark:bg-amber-400/15 dark:text-amber-300 dark:ring-amber-400/30",
  code: "bg-emerald-100 text-emerald-800 ring-emerald-600/20 dark:bg-emerald-400/15 dark:text-emerald-300 dark:ring-emerald-400/30",
};

/**
 * Snippet typography routing.
 *
 * `pdf_page` and `diagram` snippets are running prose (a PDF text layer, or a
 * VLM-generated caption); monospace at paragraph length measurably hurts
 * readability. `table` snippets are serialized markdown and `code` snippets are
 * source text -- both are column- and indentation-sensitive, so mono is load
 * bearing rather than decorative.
 *
 * Paths and ids are always mono regardless of modality; they are literal
 * identifiers, not prose.
 */
const PROSE_MODALITIES: ReadonlySet<Modality> = new Set<Modality>([
  "pdf_page",
  "diagram",
]);

export function snippetFontClass(modality: Modality): string {
  if (PROSE_MODALITIES.has(modality)) return "font-sans leading-relaxed";

  // `whitespace-pre` + per-element horizontal scroll, NOT `pre-wrap`: wrapping a
  // markdown pipe table destroys the column alignment that mono is here to
  // preserve, which defeats the point. Scrolling keeps rows intact and keeps the
  // overflow inside the card -- the page body itself never scrolls sideways.
  return "font-mono text-[0.8125rem] leading-snug whitespace-pre overflow-x-auto";
}
