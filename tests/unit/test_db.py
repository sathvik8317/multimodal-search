import pyarrow as pa
import pytest

from mmsearch import config
from mmsearch.db import ensure_fts_index, open_table, upsert
from mmsearch.schema import ARROW_SCHEMA, Modality, Row, TextSource


def _unit_vector(index: int, dim: int = config.EMBED_DIM) -> list[float]:
    vector = [0.0] * dim
    vector[index] = 1.0
    return vector


def _row(id_: str, content_text: str, vector_index: int, **overrides) -> Row:
    defaults = dict(
        id=id_,
        modality=Modality.CODE,
        content_text=content_text,
        text_source=TextSource.CODE_SOURCE,
        vector=_unit_vector(vector_index),
        source_path="src/a.py",
    )
    defaults.update(overrides)
    return Row(**defaults)


# --- open_table -----------------------------------------------------------------------

def test_open_table_creates_table_with_correct_schema(tmp_path):
    table = open_table(uri=tmp_path)
    assert table.schema == ARROW_SCHEMA
    assert table.count_rows() == 0


def test_open_table_is_idempotent(tmp_path):
    table1 = open_table(uri=tmp_path)
    upsert(table1, [_row("code:a.py#f", "hello", 0)])

    table2 = open_table(uri=tmp_path)  # re-open, should not recreate/wipe

    assert table2.count_rows() == 1


# --- upsert -----------------------------------------------------------------------------

def test_upsert_inserts_new_rows(tmp_path):
    table = open_table(uri=tmp_path)
    upsert(table, [_row("code:a.py#f", "hello world", 0), _row("code:a.py#g", "goodbye", 1)])
    assert table.count_rows() == 2


def test_upsert_empty_list_is_noop(tmp_path):
    table = open_table(uri=tmp_path)
    upsert(table, [])
    assert table.count_rows() == 0


def test_upsert_same_id_does_not_duplicate(tmp_path):
    table = open_table(uri=tmp_path)
    upsert(table, [_row("code:a.py#f", "hello world", 0)])
    upsert(table, [_row("code:a.py#f", "hello world", 0)])
    assert table.count_rows() == 1


def test_upsert_same_id_updates_content(tmp_path):
    table = open_table(uri=tmp_path)
    upsert(table, [_row("code:a.py#f", "original text", 0)])
    upsert(table, [_row("code:a.py#f", "updated text", 0)])

    assert table.count_rows() == 1
    rows = table.to_arrow().to_pylist()
    assert rows[0]["content_text"] == "updated text"


# --- ensure_fts_index / full-text search -------------------------------------------------

def test_fts_search_finds_matching_token(tmp_path):
    table = open_table(uri=tmp_path)
    upsert(
        table,
        [
            _row("code:a.py#f", "the retry backoff is exponential", 0),
            _row("code:b.py#g", "completely unrelated gardening content", 1),
        ],
    )
    ensure_fts_index(table)

    results = table.search("backoff", query_type="fts").to_list()

    assert [r["id"] for r in results] == ["code:a.py#f"]


def test_fts_search_reflects_updated_content(tmp_path):
    table = open_table(uri=tmp_path)
    upsert(table, [_row("code:a.py#f", "original searchable text", 0)])
    ensure_fts_index(table)
    upsert(table, [_row("code:a.py#f", "updated different words", 0)])
    ensure_fts_index(table)

    old_token_results = table.search("searchable", query_type="fts").to_list()
    new_token_results = table.search("updated", query_type="fts").to_list()

    assert old_token_results == []
    assert [r["id"] for r in new_token_results] == ["code:a.py#f"]


# --- vector search (brute-force, no ANN index at this corpus scale) -----------------------

def test_vector_search_finds_nearest_neighbor(tmp_path):
    table = open_table(uri=tmp_path)
    upsert(
        table,
        [
            _row("code:a.py#f", "row a", 0),
            _row("code:b.py#g", "row b", 1),
            _row("code:c.py#h", "row c", 2),
        ],
    )

    results = table.search(_unit_vector(0), query_type="vector").limit(1).to_list()

    assert results[0]["id"] == "code:a.py#f"
