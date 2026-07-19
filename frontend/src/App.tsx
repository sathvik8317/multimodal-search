import { useRef, useState } from "react";
import { search, type SearchResult } from "./api";
import { SearchBar } from "./components/SearchBar";
import { ResultsList, type Status } from "./components/ResultsList";

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);

  // Monotonic request id. Submitting again before the previous response lands
  // would otherwise let a slow earlier request overwrite a newer one -- easy to
  // hit, since reranking makes some queries noticeably slower than others.
  const latestRequest = useRef(0);

  async function runSearch() {
    const trimmed = query.trim();
    if (trimmed === "") return;

    const requestId = ++latestRequest.current;
    setStatus("loading");
    setError(null);

    try {
      const found = await search(trimmed);
      if (requestId !== latestRequest.current) return;
      setResults(found);
      setStatus("done");
    } catch (err) {
      if (requestId !== latestRequest.current) return;
      setError(err instanceof Error ? err.message : "Search failed.");
      setStatus("error");
    }
  }

  return (
    <div className="min-h-screen bg-bg font-sans text-fg">
      <div className="mx-auto flex max-w-3xl flex-col gap-8 px-4 py-10 sm:px-6 sm:py-16">
        <header className="flex flex-col gap-2">
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            Multimodal Search
          </h1>
          <p className="text-sm text-fg-muted">
            PDFs, diagrams, tables, and code in one index — hybrid retrieval
            with reciprocal rank fusion and reranking.
          </p>
        </header>

        <SearchBar
          query={query}
          onQueryChange={setQuery}
          onSubmit={runSearch}
          busy={status === "loading"}
        />

        <main>
          <ResultsList status={status} results={results} error={error} />
        </main>
      </div>
    </div>
  );
}
