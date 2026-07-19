"""Production ASGI entrypoint: `uvicorn mmsearch.api.server:app`

Wires the real search pipeline -- the real ingested LanceDB table, real
Cohere embedding + reranking (mode="rrf+rerank", the full pipeline) -- into
the FastAPI app factory from api/main.py. Loads .env for COHERE_API_KEY,
same cwd-relative convention as ingest/cli.py.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(".env"))

from mmsearch import config, db
from mmsearch.api.main import create_app
from mmsearch.clients.cohere import CohereClient
from mmsearch.retrieve.pipeline import build_search_fn

_table = db.open_table()
_client = CohereClient()
_search_fn = build_search_fn(_table, _client, _client, mode="rrf+rerank")

app = create_app(_search_fn, thumbnails_dir=config.THUMBNAILS_DIR)
