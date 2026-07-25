from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
LANCEDB_URI = DATA_DIR / "lancedb"
CAPTION_CACHE_DIR = DATA_DIR / "caption_cache"
# Where /upload stages an uploaded file before ingest_uploaded_file() reads
# it. Local/ephemeral by design -- Render's filesystem is wiped on every
# restart/redeploy, which is fine: staged source files aren't the record of
# truth, the LanceDB row (on R2) and the pushed thumbnail (on R2) are.
UPLOAD_STAGING_DIR = DATA_DIR / "upload_staging"

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

# OpenAI vision model for ApiCaptioner (clients/captioner_api.py): captions
# uploaded diagrams/scanned pages on the deployed server, which has no
# GPU/torch for the local moondream2 captioner. Local ingestion is unaffected.
OPENAI_VISION_MODEL = "gpt-4o-mini"

RERANK_MODEL = "rerank-v3.5"

# Retrieval funnel: fetch N per retriever -> RRF fuse -> rerank shortlist M -> return top-k.
FETCH_N = 50
RERANK_M = 25
TOP_K = 5

# Re-validated (not just carried over) against the new three-way (Cohere +
# OpenAI + FTS) fusion via the same controlled rrf-only k=60-vs-20 comparison
# used for the original single-space tuning, isolated to rrf-only mode (no
# reranker involved) so the effect is attributable to RRF_K alone. Result:
# aggregate hit-rate@5 is 0.920 at k=20 vs. 0.880 at k=60 on the 25-label set.
# 15/25 queries are identical at both k values; of the 10 that reorder, only
# 1 flips hit-to-miss (k=20 -> k=60), and it's the same mechanism the
# original tuning found: a code query's correct symbol sits at rank 5 at
# k=20 and gets diluted past top-5 at k=60 as the flatter k=60 weighting lets
# weaker competition in one retriever crowd out a candidate already strong in
# another. Zero queries regress in the other direction. The mechanism is
# retriever-count-agnostic, so it reproduces unchanged whether fusing 2 lists
# or 3. See EMBEDDING_MIGRATION_PLAN.md for the full per-query diff.
RRF_K = 20

# Minimum SearchResult.score a result must reach to be returned. Note the two
# scoring regimes mean different things: Cohere Rerank's relevance_score is a
# genuine confidence signal, while the RRF-fallback positional score
# 1/(rank+1) (used by vector-only/rrf-only/a failed-or-absent reranker) is
# purely positional -- the top result always scores 1.0 regardless of actual
# relevance -- so this same threshold behaves as a quality filter in rerank
# mode but as an effective top-N position cap in the RRF-fallback modes. See
# retrieve/pipeline.py's build_search_fn docstring. This is the fallback
# default; Settings.min_score_threshold (MMSEARCH_MIN_SCORE_THRESHOLD) can
# override it per-deployment without a code change.
#
# 0.10, not a guess: re-swept (rrf+rerank mode, real Cohere reranker, real
# committed 76-row index) after fixing the table-modality reranking
# miscalibration (retrieve/pipeline.py's _rerank_text -- tables were
# scoring 0.39-0.66 on completely unrelated queries pre-fix, drowning out
# any signal a score threshold could act on). That fix changed which
# threshold wins: 0.10 now halves false-positive-rate vs 0.05 (0.200 vs
# 0.400, 5 negative labels -- queries with no correct answer anywhere in
# the corpus, see eval/labels.yaml) at *identical* hit-rate@5 (0.760, 19/25
# positive labels) -- a tradeoff that didn't exist in the original sweep,
# where 0.05-0.20 were all identical because the table bug swamped
# everything the threshold could distinguish. 0.30 is still strictly worse
# (hit-rate@5 drops to 0.680, no further false-positive gain).
#
# Threshold tuning still can't fix a confidently-wrong reranker, only
# genuinely low-confidence noise -- the one remaining leak at 0.10 is an
# unrelated PDF-page confusion (not a table), scoring 0.84. See
# eval/run.py's false_positive_rate() and eval/labels.yaml's negative
# labels for the mechanism.
MIN_SCORE_THRESHOLD = 0.10

# Table ingestion: cap embedded rows so a huge CSV doesn't hard-fail on embed
# input limits or produce a semantically useless single-vector embedding.
# The full row count is still recorded in Row.metadata (truncated/total_rows).
MAX_TABLE_ROWS = 200

# Character budget for the embedded markdown, independent of MAX_TABLE_ROWS: a
# wide table (many columns) can exceed OpenAI's 8192-token input limit well
# before hitting the row cap. Validated by direct calibration against all 4
# real corpus CSVs (12-18 columns): 20000 chars was confirmed UNSAFE -- 2 of 4
# still exceeded the 8192-token limit (automobile_dataset.csv at 19958 chars,
# ecommerce_sales_analytics_5000.csv at 18860 chars both hard-failed with a
# live BadRequestError). 12000 chars cleared all 4 with margin (11873-11961
# chars each). Tokenization density varies by content, so this is a
# conservative empirical bound, not a fixed chars-per-token formula -- if a
# future corpus file still overflows at this budget, lower it further rather
# than trusting an untested chars/token ratio (EMBEDDING_MIGRATION_PLAN.md).
MAX_TABLE_EMBED_CHARS = 12000
