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


def test_embed_dim_matches_cohere_v4_default():
    assert config.EMBED_DIM == 1536


def test_retrieval_tuning_knobs_are_positive_ints():
    for value in (config.FETCH_N, config.RERANK_M, config.TOP_K, config.RRF_K):
        assert isinstance(value, int)
        assert value > 0


def test_retrieval_knobs_respect_funnel_ordering():
    # fetch -> fuse -> rerank shortlist -> final top-k, each stage narrows or holds
    assert config.RERANK_M <= config.FETCH_N
    assert config.TOP_K <= config.RERANK_M


def test_model_ids_are_nonempty_strings():
    assert isinstance(config.EMBED_MODEL, str) and config.EMBED_MODEL
    assert isinstance(config.RERANK_MODEL, str) and config.RERANK_MODEL


def test_thumbnails_dir_is_relative_to_data_dir():
    assert config.THUMBNAILS_DIR == config.DATA_DIR / "thumbnails"
