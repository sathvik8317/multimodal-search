import { useRef, useState, type FormEvent } from "react";
import { search, setApiKey, UnauthorizedError, type SearchResult } from "./api";
import { SearchBar } from "./components/SearchBar";
import { ResultsList, type Status } from "./components/ResultsList";

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [needsKey, setNeedsKey] = useState(false);
  const [keyInput, setKeyInput] = useState("");

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
      setNeedsKey(false);
    } catch (err) {
      if (requestId !== latestRequest.current) return;
      if (err instanceof UnauthorizedError) {
        // Distinct from the generic error banner: prompt for a key instead.
        setNeedsKey(true);
        setStatus("idle");
        return;
      }
      setError(err instanceof Error ? err.message : "Search failed.");
      setStatus("error");
    }
  }

  function handleKeySubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = keyInput.trim();
    if (trimmed === "") return;
    setApiKey(trimmed);
    setKeyInput("");
    setNeedsKey(false);
    void runSearch();
  }

  return (
    <div className="min-h-screen bg-bg font-sans text-fg">
      <div className="mx-auto flex max-w-3xl flex-col gap-8 px-4 py-10 sm:px-6 sm:py-16">
        <header className="flex flex-col gap-2">
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            Multimodal Search
          </h1>
          <p className="text-sm text-fg-muted">
            PDFs, diagrams, tables, and code in one index: hybrid retrieval
            with reciprocal rank fusion and reranking.
          </p>
        </header>

        <SearchBar
          query={query}
          onQueryChange={setQuery}
          onSubmit={runSearch}
          busy={status === "loading"}
        />

        {needsKey && (
          <form
            onSubmit={handleKeySubmit}
            className="flex flex-col gap-2 rounded-xl border border-border bg-surface p-4 sm:flex-row sm:items-center"
          >
            <label htmlFor="api-key" className="shrink-0 text-sm text-fg-muted">
              API key required
            </label>
            <input
              id="api-key"
              type="password"
              value={keyInput}
              onChange={(event) => setKeyInput(event.target.value)}
              placeholder="Enter MMSEARCH_API_KEY"
              autoFocus
              className="flex-1 rounded-lg border border-border bg-bg px-3 py-2 text-fg placeholder:text-fg-muted focus:ring-2 focus:ring-accent focus:outline-none"
            />
            <button
              type="submit"
              disabled={keyInput.trim() === ""}
              className="cursor-pointer rounded-lg bg-accent px-4 py-2 font-medium text-accent-fg transition-opacity duration-200 hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Save
            </button>
          </form>
        )}

        <main>
          <ResultsList status={status} results={results} error={error} />
        </main>
      </div>
    </div>
  );
}
