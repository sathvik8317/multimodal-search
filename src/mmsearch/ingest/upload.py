"""Single-file ingestion for the authenticated /upload endpoint.

Thin orchestration over the same primitives ingest_corpus uses for a full
directory walk -- classify_file, _ingest_one, db.upsert (see ingest/base.py).
Adds the concerns a directory walk never needed because its input is already
trusted: content-vs-extension validation (an upload is untrusted input) and
uploader/uploaded_at provenance tagging (so a shared index stays moderable --
see UPLOAD_PLAN.md's "shared + tagged" content model decision).

HTTP concerns (multipart parsing, streamed size cap, rate limiting) are not
this module's job; api/main.py's /upload route is a thin wrapper around
ingest_uploaded_file() that maps its exceptions to HTTP responses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lancedb.table import Table

from mmsearch import db
from mmsearch.clients.protocols import Captioner, Embedders
from mmsearch.ingest.base import _ingest_one, classify_file
from mmsearch.ingest.validation import validate_upload_content
from mmsearch.schema import Row

_ANON_UPLOADER = "anon"
_SAFE_UPLOADER_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_UPLOADER_LEN = 64


class UnsupportedUploadError(ValueError):
    """Raised when classify_file doesn't recognize the upload's extension."""


@dataclass
class UploadResult:
    modality: str
    rows_written: int
    thumbnail_refs: list[str]


def _sanitize_filename(filename: str) -> str:
    # Path(...).name strips any directory components (including ../ traversal
    # and absolute-path segments) -- the untrusted filename never gets to
    # choose where on disk it lands.
    name = Path(filename).name.strip()
    return name or "upload"


def _sanitize_uploader(uploader: str | None) -> str:
    if not uploader or not uploader.strip():
        return _ANON_UPLOADER
    slug = _SAFE_UPLOADER_RE.sub("-", uploader.strip())[:_MAX_UPLOADER_LEN].strip("-")
    return slug or _ANON_UPLOADER


def _stamp_provenance(rows: list[Row], uploader: str, uploaded_at: str) -> None:
    for row in rows:
        meta = json.loads(row.metadata)
        meta["uploader"] = uploader
        meta["uploaded_at"] = uploaded_at
        row.metadata = json.dumps(meta)


def ingest_uploaded_file(
    filename: str,
    data: bytes,
    uploader: str | None,
    embedders: Embedders,
    captioner: Captioner,
    table: Table,
    staging_root: Path,
    thumbnails_dir: Path,
) -> UploadResult:
    safe_name = _sanitize_filename(filename)
    safe_uploader = _sanitize_uploader(uploader)

    dest_path = staging_root / "uploads" / safe_uploader / safe_name

    category = classify_file(dest_path)
    if category is None:
        raise UnsupportedUploadError(f"unsupported extension {dest_path.suffix!r}")

    validate_upload_content(dest_path.suffix, data)  # raises UploadValidationError

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(data)

    rows = _ingest_one(category, dest_path, staging_root, embedders, captioner, thumbnails_dir)
    uploaded_at = datetime.now(timezone.utc).isoformat()
    _stamp_provenance(rows, safe_uploader, uploaded_at)

    db.upsert(table, rows)

    modality = rows[0].modality.value if rows else category
    thumbnail_refs = [row.thumbnail_ref for row in rows if row.thumbnail_ref]
    return UploadResult(modality=modality, rows_written=len(rows), thumbnail_refs=thumbnail_refs)
