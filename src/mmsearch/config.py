from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
LANCEDB_URI = DATA_DIR / "lancedb"
CAPTION_CACHE_DIR = DATA_DIR / "caption_cache"

TABLE_NAME = "chunks"

# Cohere Embed v4 default output dimension (Matryoshka: 256/512/1024/1536 also supported).
EMBED_DIM = 1536

EMBED_MODEL = "embed-v4.0"
RERANK_MODEL = "rerank-v3.5"

# Retrieval funnel: fetch N per retriever -> RRF fuse -> rerank shortlist M -> return top-k.
FETCH_N = 50
RERANK_M = 25
TOP_K = 5
RRF_K = 60
