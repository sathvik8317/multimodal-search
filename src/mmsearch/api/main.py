"""FastAPI app wiring the search pipeline behind HTTP endpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from mmsearch import config
from mmsearch.retrieve.types import SearchFn

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(search_fn: SearchFn, thumbnails_dir: Path = config.THUMBNAILS_DIR) -> FastAPI:
    app = FastAPI(title="Multimodal Search")

    thumbnails_dir = Path(thumbnails_dir)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/search")
    def search(q: str, k: int = config.TOP_K) -> list[dict]:
        results = search_fn(q, k=k)
        return [asdict(result) for result in results]

    app.mount("/thumbnails", StaticFiles(directory=thumbnails_dir), name="thumbnails")
    app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")

    return app
