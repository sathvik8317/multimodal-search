from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
LANCEDB_URI = DATA_DIR / "lancedb"
CAPTION_CACHE_DIR = DATA_DIR / "caption_cache"

TABLE_NAME = "chunks"

# Cohere Embed v4 default output dimension (Matryoshka: 256/512/1024/1536 also
# supported). Cohere embeds pdf_page/diagram *images* only (see
# EMBEDDING_MIGRATION_PLAN.md); text embedding moved to OpenAI below.
COHERE_EMBED_MODEL = "embed-v4.0"
COHERE_EMBED_DIM = 1536

# OpenAI text-embedding-3-small: table/code/caption *text* embedding. Dimension
# confirmed live against the real API, not taken from docs on faith.
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_EMBED_DIM = 1536

RERANK_MODEL = "rerank-v3.5"

# Retrieval funnel: fetch N per retriever -> RRF fuse -> rerank shortlist M -> return top-k.
FETCH_N = 50
RERANK_M = 25
TOP_K = 5

# Tuned against the OLD single-space (Cohere-only) index via a controlled
# rrf-only comparison at k=60 vs. k=20 (see prior commit history for the
# methodology). The move to a two-space (Cohere + OpenAI), three-way RRF
# fusion invalidates that tuning -- retune against the re-ingested index and
# replace this comment with the new evidence (EMBEDDING_MIGRATION_PLAN.md §8).
RRF_K = 20

# Table ingestion: cap embedded rows so a huge CSV doesn't hard-fail on embed
# input limits or produce a semantically useless single-vector embedding.
# The full row count is still recorded in Row.metadata (truncated/total_rows).
MAX_TABLE_ROWS = 200

# Character budget for the embedded markdown, independent of MAX_TABLE_ROWS: a
# wide table (many columns) can exceed OpenAI's 8192-token input limit well
# before hitting the row cap (see EMBEDDING_MIGRATION_PLAN.md -- confirmed
# live against the real corpus). Placeholder pending empirical validation.
MAX_TABLE_EMBED_CHARS = 20000
