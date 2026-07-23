"""FastAPI app wiring the search pipeline behind HTTP endpoints."""

from __future__ import annotations

import mimetypes
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from lancedb.table import Table

from mmsearch import config
from mmsearch.api.deps import rate_limit, require_api_key, upload_rate_limit
from mmsearch.clients.protocols import Captioner, Embedders
from mmsearch.ingest.upload import UnsupportedUploadError, ingest_uploaded_file
from mmsearch.ingest.validation import UploadValidationError
from mmsearch.retrieve.types import SearchFn
from mmsearch.settings import Settings, get_settings

_STATIC_DIR = Path(__file__).resolve().parent / "static"

_UPLOADS_PREFIX = "uploads/"
_UPLOAD_CHUNK_BYTES = 65536
_DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class ThumbnailStorage(Protocol):
    def get_bytes(self, key: str) -> bytes: ...
    def put_bytes(self, key: str, data: bytes) -> None: ...


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read file in chunks, rejecting once the total exceeds max_bytes.

    Starlette's multipart parser has already spooled the full body by the
    time this runs (FastAPI/Starlette has no lower-level streaming hook), so
    this doesn't stop an oversized *transfer* -- but it does stop our own
    code (ingest, embedding, storage writes) from ever running on more than
    max_bytes of data.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="file too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _resolve_thumbnail(thumbnails_root: Path, thumb_path: str) -> Path:
    """Join and containment-check a request path against thumbnails_root.

    Trust boundary: StaticFiles used to do this for free. Joining an absolute
    path or a `..`/`..\\` segment onto a Path can still land outside
    thumbnails_root (pathlib lets an absolute operand discard the base
    entirely), so containment is verified on the *resolved* path, not the
    joined-but-unresolved one.
    """
    candidate = (thumbnails_root / thumb_path).resolve()
    if not candidate.is_relative_to(thumbnails_root) or not candidate.is_file():
        raise HTTPException(status_code=404)
    return candidate


def create_app(
    search_fn: SearchFn,
    thumbnails_dir: Path = config.THUMBNAILS_DIR,
    settings: Settings | None = None,
    upload_thumbnail_storage: ThumbnailStorage | None = None,
    upload_table: Table | None = None,
    upload_embedders: Embedders | None = None,
    upload_captioner: Captioner | None = None,
    upload_staging_root: Path | None = None,
    max_upload_bytes: int = _DEFAULT_MAX_UPLOAD_BYTES,
) -> FastAPI:
    app = FastAPI(title="Multimodal Search")

    settings = settings or get_settings()
    # CORS setup below (construction-time) and the require_api_key/rate_limit
    # dependencies (request-time) must agree on one Settings instance; this
    # override is what makes that instance injectable in tests.
    app.dependency_overrides[get_settings] = lambda: settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key"],
    )

    thumbnails_dir = Path(thumbnails_dir).resolve()
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    @app.get("/search", dependencies=[Depends(require_api_key), Depends(rate_limit)])
    def search(q: str, k: int = config.TOP_K) -> list[dict]:
        results = search_fn(q, k=k)
        return [asdict(result) for result in results]

    @app.get("/thumbnails/{thumb_path:path}", dependencies=[Depends(require_api_key)])
    def get_thumbnail(thumb_path: str) -> Response:
        if thumb_path.startswith(_UPLOADS_PREFIX):
            if upload_thumbnail_storage is None:
                raise HTTPException(status_code=404)
            try:
                data = upload_thumbnail_storage.get_bytes(thumb_path)
            except FileNotFoundError:
                raise HTTPException(status_code=404) from None
            media_type = mimetypes.guess_type(thumb_path)[0] or "application/octet-stream"
            return Response(content=data, media_type=media_type)
        return FileResponse(_resolve_thumbnail(thumbnails_dir, thumb_path))

    upload_enabled = (
        upload_table is not None
        and upload_embedders is not None
        and upload_captioner is not None
        and upload_staging_root is not None
    )
    if upload_enabled:
        upload_staging_root = Path(upload_staging_root).resolve()
        upload_staging_root.mkdir(parents=True, exist_ok=True)

        @app.post("/upload", dependencies=[Depends(require_api_key), Depends(upload_rate_limit)])
        async def upload(
            file: UploadFile = File(...), uploader: str | None = Form(None)
        ) -> dict:
            data = await _read_limited(file, max_upload_bytes)
            try:
                result = ingest_uploaded_file(
                    filename=file.filename or "upload",
                    data=data,
                    uploader=uploader,
                    embedders=upload_embedders,
                    captioner=upload_captioner,
                    table=upload_table,
                    staging_root=upload_staging_root,
                    thumbnails_dir=thumbnails_dir,
                )
            except (UnsupportedUploadError, UploadValidationError) as exc:
                raise HTTPException(status_code=415, detail=str(exc)) from exc
            except Exception as exc:  # noqa: BLE001 -- a bad upload must not 500 the server
                raise HTTPException(status_code=422, detail=f"ingest failed: {exc}") from exc

            if upload_thumbnail_storage is not None:
                for ref in result.thumbnail_refs:
                    local_path = thumbnails_dir / ref
                    if local_path.is_file():
                        upload_thumbnail_storage.put_bytes(ref, local_path.read_bytes())

            upload_table.checkout_latest()

            return {
                "status": "ok",
                "filename": file.filename,
                "modality": result.modality,
                "rows_written": result.rows_written,
            }

    app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")

    return app
