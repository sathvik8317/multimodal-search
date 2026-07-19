import { useState } from "react";
import type { SearchResult } from "../api";
import { snippetFontClass } from "../modality";
import { ModalityBadge } from "./ModalityBadge";

/** Lucide `file-text`. Placeholder for results with no thumbnail. */
function FileIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
      <path d="M14 2v4a2 2 0 0 0 2 2h4" />
      <path d="M10 9H8" />
      <path d="M16 13H8" />
      <path d="M16 17H8" />
    </svg>
  );
}

/**
 * pdf_page thumbnail refs embed a '#' -- make_id() in schema.py builds them as
 * "specs/paper.pdf#p5.png". Interpolated raw into a URL, '#' opens a fragment
 * and the browser silently requests "/thumbnails/specs/paper.pdf", which 404s.
 * (The previous vanilla UI had this bug too; every PDF thumbnail was broken.)
 * Encode per path segment so '/' separators survive but '#' becomes %23.
 */
function thumbnailUrl(ref: string): string {
  return `/thumbnails/${ref.split("/").map(encodeURIComponent).join("/")}`;
}

function Thumbnail({ result }: { result: SearchResult }) {
  // A thumbnail_ref can be absent (empty string is the backend default) or
  // point at a file that failed to generate -- handle both, rather than
  // rendering a broken-image glyph.
  const [failed, setFailed] = useState(false);

  if (!result.thumbnail_ref || failed) {
    return (
      <div className="flex size-20 shrink-0 items-center justify-center rounded-md border border-border bg-bg">
        <FileIcon className="size-7 text-fg-muted" />
      </div>
    );
  }

  return (
    <img
      src={thumbnailUrl(result.thumbnail_ref)}
      alt=""
      loading="lazy"
      onError={() => setFailed(true)}
      className="size-20 shrink-0 rounded-md border border-border bg-bg object-cover"
    />
  );
}

export function ResultCard({ result }: { result: SearchResult }) {
  return (
    <article className="flex gap-4 rounded-xl border border-border bg-surface p-4 transition-colors duration-200 hover:border-fg-muted">
      <Thumbnail result={result} />

      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <ModalityBadge modality={result.modality} />
          {/* Identifiers are always mono, regardless of modality. */}
          <span className="truncate font-mono text-xs text-fg-muted">
            {result.source_path}
          </span>
        </div>

        <p className={`m-0 text-fg ${snippetFontClass(result.modality)}`}>
          {result.snippet}
        </p>

        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-xs text-fg-muted">
          <span>score {result.score.toFixed(3)}</span>
          <span aria-hidden="true">·</span>
          <span className="truncate">{result.id}</span>
        </div>
      </div>
    </article>
  );
}
