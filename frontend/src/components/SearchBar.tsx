import { type FormEvent } from "react";

/** Lucide `search`. Inline SVG, not an emoji (checklist). */
function SearchIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

interface SearchBarProps {
  query: string;
  onQueryChange: (value: string) => void;
  onSubmit: () => void;
  busy: boolean;
}

export function SearchBar({
  query,
  onQueryChange,
  onSubmit,
  busy,
}: SearchBarProps) {
  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    onSubmit();
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-2 sm:flex-row">
      <div className="relative flex-1">
        <SearchIcon className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-fg-muted" />
        <input
          type="search"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Search PDFs, diagrams, tables, and code…"
          aria-label="Search query"
          autoFocus
          className="w-full rounded-lg border border-border bg-surface py-2.5 pr-3 pl-9 text-fg placeholder:text-fg-muted focus:ring-2 focus:ring-accent focus:outline-none"
        />
      </div>
      <button
        type="submit"
        disabled={busy || query.trim() === ""}
        className="cursor-pointer rounded-lg bg-accent px-5 py-2.5 font-medium text-accent-fg transition-opacity duration-200 hover:opacity-90 focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-bg focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
      >
        {busy ? "Searching…" : "Search"}
      </button>
    </form>
  );
}
