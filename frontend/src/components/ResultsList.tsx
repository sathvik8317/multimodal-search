import type { ReactNode } from "react";
import type { SearchResult } from "../api";
import { ResultCard } from "./ResultCard";

export type Status = "idle" | "loading" | "error" | "done";

function SkeletonCard() {
  return (
    <div className="flex animate-pulse gap-4 rounded-xl border border-border bg-surface p-4">
      <div className="size-20 shrink-0 rounded-md bg-border" />
      <div className="flex flex-1 flex-col gap-2.5 py-1">
        <div className="h-4 w-24 rounded bg-border" />
        <div className="h-3 w-full rounded bg-border" />
        <div className="h-3 w-4/5 rounded bg-border" />
      </div>
    </div>
  );
}

function Notice({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-xl border border-dashed border-border px-4 py-10 text-center text-sm text-fg-muted">
      {children}
    </p>
  );
}

interface ResultsListProps {
  status: Status;
  results: SearchResult[];
  error: string | null;
}

export function ResultsList({ status, results, error }: ResultsListProps) {
  if (status === "idle") {
    return (
      <Notice>
        Try <em>“how does reciprocal rank fusion work”</em> and watch one query
        hit four modalities.
      </Notice>
    );
  }

  if (status === "loading") {
    return (
      // aria-busy + role=status so a screen reader announces the pending state
      // instead of silence.
      <div
        className="flex flex-col gap-3"
        role="status"
        aria-busy="true"
        aria-label="Searching"
      >
        <SkeletonCard />
        <SkeletonCard />
        <SkeletonCard />
      </div>
    );
  }

  if (status === "error") {
    return <Notice>{error ?? "Something went wrong."}</Notice>;
  }

  if (results.length === 0) {
    return <Notice>No results. Try different wording.</Notice>;
  }

  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-fg-muted" role="status">
        {results.length} result{results.length === 1 ? "" : "s"}
      </p>
      {results.map((result) => (
        <ResultCard key={result.id} result={result} />
      ))}
    </div>
  );
}
