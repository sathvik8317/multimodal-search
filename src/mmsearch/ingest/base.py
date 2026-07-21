"""Corpus walker: recurses a directory, dispatches files by extension to the
per-modality ingesters, and writes the resulting rows to LanceDB.

Extension routing reflects what the ingesters actually support today, not
PLAN.md's aspirational scope: ingest_table only parses CSV (xlsx was
deliberately deprioritized during Phase 1), so .xlsx files are treated as
unsupported and recorded in IngestStats.skipped rather than silently
mis-ingested or crashing the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lancedb.table import Table

from mmsearch import config, db
from mmsearch.clients.protocols import Captioner, Embedders
from mmsearch.ingest.code import ingest_code_file
from mmsearch.ingest.documents import ingest_diagram, ingest_pdf
from mmsearch.ingest.tables import ingest_table
from mmsearch.schema import Row

_DIAGRAM_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def classify_file(path: Path) -> str | None:
    """Return a coarse category name for a file, or None if unsupported."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in _DIAGRAM_EXTENSIONS:
        return "diagram"
    if suffix == ".csv":
        return "table"
    if suffix == ".py":
        return "code"
    return None


def walk_corpus(root: Path) -> list[Path]:
    """All non-hidden files under root, deterministically (sorted) ordered."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


@dataclass
class IngestStats:
    rows_written: int = 0
    rows_by_modality: dict[str, int] = field(default_factory=dict)
    rows_by_text_source: dict[str, int] = field(default_factory=dict)
    files_processed: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (relpath, reason)

    def record(self, rows: list[Row]) -> None:
        for row in rows:
            self.rows_written += 1
            modality = row.modality.value
            self.rows_by_modality[modality] = self.rows_by_modality.get(modality, 0) + 1
            text_source = row.text_source.value
            self.rows_by_text_source[text_source] = self.rows_by_text_source.get(text_source, 0) + 1


def ingest_corpus(
    root: Path,
    embedders: Embedders,
    captioner: Captioner,
    table: Table,
    thumbnails_dir: Path = config.THUMBNAILS_DIR,
) -> IngestStats:
    """Walk root, ingest every supported file, and upsert its rows into table.

    A single file failing to ingest (e.g. a malformed PDF) does not abort the
    run -- it's recorded in stats.skipped with the exception message, and
    rows already written for earlier files remain persisted (upsert happens
    per-file, not batched at the end).
    """
    stats = IngestStats()
    for path in walk_corpus(root):
        relpath = path.relative_to(root).as_posix()
        category = classify_file(path)
        if category is None:
            stats.skipped.append((relpath, f"unsupported extension {path.suffix!r}"))
            continue

        try:
            rows = _ingest_one(category, path, root, embedders, captioner, thumbnails_dir)
        except Exception as exc:  # noqa: BLE001 -- one bad file must not abort the whole corpus
            stats.skipped.append((relpath, f"{type(exc).__name__}: {exc}"))
            continue

        stats.files_processed += 1
        db.upsert(table, rows)
        stats.record(rows)
    return stats


def _ingest_one(
    category: str,
    path: Path,
    root: Path,
    embedders: Embedders,
    captioner: Captioner,
    thumbnails_dir: Path,
) -> list[Row]:
    if category == "pdf":
        return ingest_pdf(path, root, embedders, captioner, thumbnails_dir=thumbnails_dir)
    if category == "diagram":
        return [ingest_diagram(path, root, embedders, captioner, thumbnails_dir=thumbnails_dir)]
    if category == "table":
        return [ingest_table(path, root, embedders.text)]
    if category == "code":
        return ingest_code_file(path, root, embedders.text)
    raise ValueError(f"unknown category: {category!r}")
