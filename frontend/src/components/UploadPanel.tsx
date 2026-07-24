import { useRef, useState, type FormEvent } from "react";
import { uploadFile, UnauthorizedError, type UploadResponse } from "../api";
import { MODALITY_LABEL } from "../modality";

type UploadStatus = "idle" | "uploading" | "done" | "error";

const ACCEPTED_EXTENSIONS =
  ".pdf,.png,.jpg,.jpeg,.gif,.bmp,.webp,.csv,.xlsx,.py";

interface UploadPanelProps {
  onUnauthorized: () => void;
}

export function UploadPanel({ onUnauthorized }: UploadPanelProps) {
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploaderName, setUploaderName] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;

    setStatus("uploading");
    setError(null);
    setResult(null);

    try {
      const response = await uploadFile(file, uploaderName);
      setResult(response);
      setStatus("done");
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      if (err instanceof UnauthorizedError) {
        onUnauthorized();
        setStatus("idle");
        return;
      }
      setError(err instanceof Error ? err.message : "Upload failed.");
      setStatus("error");
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-3 rounded-xl border border-border bg-surface p-4"
    >
      <div className="flex flex-col gap-1">
        <h2 className="text-sm font-medium text-fg">Add a file to the index</h2>
        <p className="text-xs text-fg-muted">
          PDF, image, code (.py), CSV, or Excel (.xlsx).
        </p>
      </div>

      <div className="flex flex-col gap-2 sm:flex-row">
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS}
          aria-label="File to upload"
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
          disabled={status === "uploading"}
          className="cursor-pointer rounded-lg bg-accent px-4 py-2 text-sm font-medium text-accent-fg transition-opacity duration-200 hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {status === "uploading" ? "Uploading…" : "Upload"}
        </button>
      </div>

      {status === "done" && result && (
        <p className="text-xs text-fg-muted" role="status">
          Added <span className="font-medium text-fg">{result.filename}</span>{" "}
          as {MODALITY_LABEL[result.modality]} ({result.rows_written} chunk
          {result.rows_written === 1 ? "" : "s"}).
        </p>
      )}
      {status === "error" && (
        <p className="text-xs text-red-600 dark:text-red-400" role="alert">
          {error}
        </p>
      )}
    </form>
  );
}
