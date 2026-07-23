"""Production ASGI entrypoint: `uvicorn mmsearch.api.server:app`

Wires the real search pipeline -- the real ingested LanceDB table, real
Cohere image embedding + OpenAI text embedding + Cohere reranking
(mode="rrf+rerank", the full pipeline) -- into the FastAPI app factory from
api/main.py. get_settings() reads .env the same cwd-relative convention
load_dotenv() used to, and fails fast with a clear error if MMSEARCH_API_KEY
isn't configured (see settings.py).
"""

from __future__ import annotations

from mmsearch import config, db
from mmsearch.api.main import create_app
from mmsearch.clients.cohere import CohereClient
from mmsearch.clients.openai import OpenAIClient
from mmsearch.retrieve.pipeline import build_search_fn
from mmsearch.settings import get_settings
from mmsearch.storage.r2 import R2Storage

_settings = get_settings()
_table = db.open_table(
    uri=_settings.lancedb_uri or config.LANCEDB_URI,
    storage_options=_settings.r2_storage_options(),
    # An explicit lancedb_uri (R2 in production) must fail loud if the table
    # is missing -- silently creating an empty table there would serve zero
    # results with no error (see DEPLOYMENT_PLAN.md §2). The local default
    # path keeps auto-create, since that's what a fresh dev/CLI run needs.
    create_if_missing=_settings.lancedb_uri is None,
)
_cohere_client = CohereClient()
_openai_client = OpenAIClient()
_search_fn = build_search_fn(
    _table, _cohere_client, _openai_client, _cohere_client, mode="rrf+rerank"
)

# Uploaded-file thumbnails only; the curated corpus's thumbnails always stay
# local/in-git and are served by create_app's existing local-FS path. Absent
# r2_bucket (no R2 configured), uploaded-thumbnail requests 404 -- but so does
# /upload itself in that case (see api/main.py), so this is consistent.
_upload_thumbnail_storage = R2Storage(_settings.r2_bucket, settings=_settings) if _settings.r2_bucket else None

app = create_app(
    _search_fn,
    thumbnails_dir=config.THUMBNAILS_DIR,
    settings=_settings,
    upload_thumbnail_storage=_upload_thumbnail_storage,
)
