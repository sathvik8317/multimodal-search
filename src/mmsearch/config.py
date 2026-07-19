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

# Validated against the 25-label eval set (eval/labels.yaml), not a guess:
# a controlled rrf-only vs. rrf-only comparison at k=60 vs. k=20 on the real
# ingested index showed k=20 fixes 2 code queries that were missing the
# correct symbol's top-5 slot entirely at k=60 (moved to ranks 4-5), with
# zero measured regression across pdf_page/diagram/table modalities or the
# rrf+rerank mode (identical per-query hit/miss and identical top-5 order
# for every other query at both k values). Smaller k sharpens RRF's
# rank-based scoring, which helps a candidate that's already strong in one
# retriever (here: vector) but was getting diluted by weak competition in
# the other (FTS) under the flatter k=60 weighting.
RRF_K = 20

# Table ingestion: cap embedded rows so a huge CSV doesn't hard-fail on embed
# input limits or produce a semantically useless single-vector embedding.
# The full row count is still recorded in Row.metadata (truncated/total_rows).
MAX_TABLE_ROWS = 200
