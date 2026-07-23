"""FastAPI app wiring the search pipeline behind HTTP endpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from mmsearch import config
from mmsearch.api.deps import rate_limit, require_api_key
from mmsearch.retrieve.types import SearchFn
from mmsearch.settings import Settings, get_settings

_STATIC_DIR = Path(__file__).resolve().parent / "static"


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
        allow_methods=["GET"],
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
    def get_thumbnail(thumb_path: str) -> FileResponse:
        return FileResponse(_resolve_thumbnail(thumbnails_dir, thumb_path))

    app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")

    return app
