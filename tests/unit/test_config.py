from pathlib import Path

from mmsearch import config


def test_data_dir_is_a_path():
    assert isinstance(config.DATA_DIR, Path)


def test_thumbnails_dir_is_a_path():
    assert isinstance(config.THUMBNAILS_DIR, Path)


def test_lancedb_uri_is_a_path():
    assert isinstance(config.LANCEDB_URI, Path)


def test_table_name():
    assert config.TABLE_NAME == "chunks"


def test_cohere_embed_dim_matches_cohere_v4_default():
    assert config.COHERE_EMBED_DIM == 1536


def test_openai_embed_dim_matches_text_embedding_3_small():
    assert config.OPENAI_EMBED_DIM == 1536


def test_retrieval_tuning_knobs_are_positive_ints():
    for value in (config.FETCH_N, config.RERANK_M, config.TOP_K, config.RRF_K):
        assert isinstance(value, int)
        assert value > 0


def test_retrieval_knobs_respect_funnel_ordering():
    # fetch -> fuse -> rerank shortlist -> final top-k, each stage narrows or holds
    assert config.RERANK_M <= config.FETCH_N
    assert config.TOP_K <= config.RERANK_M


def test_model_ids_are_nonempty_strings():
    assert isinstance(config.COHERE_EMBED_MODEL, str) and config.COHERE_EMBED_MODEL
    assert isinstance(config.OPENAI_EMBED_MODEL, str) and config.OPENAI_EMBED_MODEL
    assert isinstance(config.RERANK_MODEL, str) and config.RERANK_MODEL


def test_thumbnails_dir_is_relative_to_data_dir():
    assert config.THUMBNAILS_DIR == config.DATA_DIR / "thumbnails"


def test_max_table_embed_chars_is_a_positive_int_under_the_row_cap_estimate():
    assert isinstance(config.MAX_TABLE_EMBED_CHARS, int)
    assert config.MAX_TABLE_EMBED_CHARS > 0


def test_max_table_embed_chars_matches_calibrated_safe_value():
    # 12000 was validated by direct calibration against all 4 real corpus
    # CSVs (12-18 columns): 20000 was confirmed UNSAFE (2 of 4 exceeded
    # OpenAI's 8192-token limit with a live BadRequestError); 12000 cleared
    # all 4 with margin. See EMBEDDING_MIGRATION_PLAN.md and config.py's
    # comment for the exact figures. If this value ever needs to change,
    # re-calibrate against real files rather than trusting a chars/token
    # formula -- tokenization density varies by content.
    assert config.MAX_TABLE_EMBED_CHARS == 12000
