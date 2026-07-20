"""Production ASGI entrypoint: `uvicorn mmsearch.api.server:app`

Wires the real search pipeline -- the real ingested LanceDB table, real
Cohere embedding + reranking (mode="rrf+rerank", the full pipeline) -- into
the FastAPI app factory from api/main.py. get_settings() reads .env the same
cwd-relative convention load_dotenv() used to, and fails fast with a clear
error if MMSEARCH_API_KEY isn't configured (see settings.py).
"""

from __future__ import annotations

from mmsearch import config, db
from mmsearch.api.main import create_app
from mmsearch.clients.cohere import CohereClient
from mmsearch.retrieve.pipeline import build_search_fn
from mmsearch.settings import get_settings

_settings = get_settings()
_table = db.open_table()
_client = CohereClient()
_search_fn = build_search_fn(_table, _client, _client, mode="rrf+rerank")

app = create_app(_search_fn, thumbnails_dir=config.THUMBNAILS_DIR, settings=_settings)
