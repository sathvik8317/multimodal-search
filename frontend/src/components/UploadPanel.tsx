import { useRef, useState, type FormEvent } from "react";
import { uploadFile, UnauthorizedError, type UploadResponse } from "../api";
import { MODALITY_LABEL } from "../modality";

type Phase = "idle" | "uploading" | "done";

interface FileOutcome {
  filename: string;
  ok: boolean;
  detail: string;
}

const ACCEPTED_EXTENSIONS =
  ".pdf,.png,.jpg,.jpeg,.gif,.bmp,.webp,.csv,.xlsx,.py";

interface UploadPanelProps {
  onUnauthorized: () => void;
}

function describe(response: UploadResponse): string {
  return `${MODALITY_LABEL[response.modality]} (${response.rows_written} chunk${
    response.rows_written === 1 ? "" : "s"
  })`;
}

export function UploadPanel({ onUnauthorized }: UploadPanelProps) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [total, setTotal] = useState(0);
  const [completed, setCompleted] = useState(0);
  const [currentFilename, setCurrentFilename] = useState<string | null>(null);
  const [outcomes, setOutcomes] = useState<FileOutcome[]>([]);
  const [uploaderName, setUploaderName] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const files = fileInputRef.current?.files;
    if (!files || files.length === 0) return;

    setPhase("uploading");
    setTotal(files.length);
    setCompleted(0);
    setOutcomes([]);

    const results: FileOutcome[] = [];

    for (const file of Array.from(files)) {
      setCurrentFilename(file.name);
      try {
        const response = await uploadFile(file, uploaderName);
        results.push({ filename: file.name, ok: true, detail: describe(response) });
      } catch (err) {
        if (err instanceof UnauthorizedError) {
          onUnauthorized();
          setPhase("idle");
          return;
        }
        results.push({
          filename: file.name,
          ok: false,
          detail: err instanceof Error ? err.message : "Upload failed.",
        });
      }
      setCompleted((n) => n + 1);
      setOutcomes([...results]);
    }

    setCurrentFilename(null);
    setPhase("done");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  const percent = total > 0 ? Math.round((completed / total) * 100) : 0;

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-3 rounded-xl border border-border bg-surface p-4"
    >
      <div className="flex flex-col gap-1">
        <h2 className="text-sm font-medium text-fg">Add files to the index</h2>
        <p className="text-xs text-fg-muted">
          PDF, image, code (.py), CSV, or Excel (.xlsx). Select multiple files at once.
        </p>
      </div>

      <div className="flex flex-col gap-2 sm:flex-row">
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={ACCEPTED_EXTENSIONS}
          aria-label="Files to upload"
          className="flex-1 rounded-lg border border-border bg-bg px-3 py-2 text-sm text-fg file:mr-3 file:cursor-pointer file:rounded-md file:border-0 file:bg-accent file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-accent-fg"
        />
        <input
          type="text"
          value={uploaderName}
          onChange={(event) => setUploaderName(event.target.value)}
          placeholder="Your name (optional)"
          aria-label="Uploader name"
          className="rounded-lg border border-border bg-bg px-3 py-2 text-sm text-fg placeholder:text-fg-muted focus:ring-2 focus:ring-accent focus:outline-none sm:w-40"
        />
        <button
          type="submit"
          disabled={phase === "uploading"}
          className="cursor-pointer rounded-lg bg-accent px-4 py-2 text-sm font-medium text-accent-fg transition-opacity duration-200 hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {phase === "uploading" ? "Uploading…" : "Upload"}
        </button>
      </div>

      {phase === "uploading" && (
        <div className="flex flex-col gap-1.5" role="status" aria-live="polite">
          <div className="flex items-center justify-between text-xs text-fg-muted">
            <span>
              {completed} of {total} uploaded
              {currentFilename ? ` — ${currentFilename}…` : ""}
            </span>
            <span>{percent}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-border">
            <div
              className="h-full rounded-full bg-accent transition-all duration-200"
              style={{ width: `${percent}%` }}
            />
          </div>
        </div>
      )}

      {phase === "done" && outcomes.length > 0 && (
        <ul className="flex flex-col gap-1 text-xs">
          {outcomes.map((outcome, index) => (
            <li
              key={`${outcome.filename}-${index}`}
              role={outcome.ok ? "status" : "alert"}
              className={
                outcome.ok
                  ? "text-fg-muted"
                  : "text-red-600 dark:text-red-400"
              }
            >
              <span className="font-medium text-fg">{outcome.filename}</span>
              {outcome.ok ? " — added as " : " — failed: "}
              {outcome.detail}
            </li>
          ))}
        </ul>
      )}
    </form>
  );
}
