import json

from mmsearch import config
from mmsearch.clients.fakes import FakeEmbeddingClient
from mmsearch.ingest.tables import ingest_table
from mmsearch.schema import Modality, TextSource
from tests.fixtures import CORPUS_DIR


def test_ingest_table_returns_row_with_correct_id():
    csv_path = CORPUS_DIR / "data" / "latency.csv"
    row = ingest_table(csv_path, CORPUS_DIR, FakeEmbeddingClient())
    assert row.id == "tbl:data/latency.csv"


def test_ingest_table_sets_modality_and_text_source():
    csv_path = CORPUS_DIR / "data" / "latency.csv"
    row = ingest_table(csv_path, CORPUS_DIR, FakeEmbeddingClient())
    assert row.modality == Modality.TABLE
    assert row.text_source == TextSource.TABLE_MARKDOWN


def test_ingest_table_content_text_contains_headers_and_looks_like_markdown():
    csv_path = CORPUS_DIR / "data" / "latency.csv"
    row = ingest_table(csv_path, CORPUS_DIR, FakeEmbeddingClient())
    assert "endpoint" in row.content_text
    assert "p50_ms" in row.content_text
    assert "p95_ms" in row.content_text
    assert "p99_ms" in row.content_text
    lines = row.content_text.strip().splitlines()
    assert lines[0].startswith("|")
    assert set(lines[1].replace("|", "").strip()) <= {"-", " "}


def test_ingest_table_vector_has_correct_dimension():
    csv_path = CORPUS_DIR / "data" / "latency.csv"
    row = ingest_table(csv_path, CORPUS_DIR, FakeEmbeddingClient())
    assert len(row.vector) == config.EMBED_DIM


def test_ingest_table_thumbnail_ref_is_empty():
    csv_path = CORPUS_DIR / "data" / "latency.csv"
    row = ingest_table(csv_path, CORPUS_DIR, FakeEmbeddingClient())
    assert row.thumbnail_ref == ""


def test_ingest_table_source_path_is_corpus_relative():
    csv_path = CORPUS_DIR / "data" / "latency.csv"
    row = ingest_table(csv_path, CORPUS_DIR, FakeEmbeddingClient())
    assert row.source_path == "data/latency.csv"


def test_ingest_table_metadata_has_correct_row_and_column_counts():
    csv_path = CORPUS_DIR / "data" / "latency.csv"
    row = ingest_table(csv_path, CORPUS_DIR, FakeEmbeddingClient())
    metadata = json.loads(row.metadata)
    assert metadata["n_rows"] == 3
    assert metadata["n_cols"] == 4
    assert metadata["columns"] == ["endpoint", "p50_ms", "p95_ms", "p99_ms"]


def test_ingest_table_is_deterministic():
    csv_path = CORPUS_DIR / "data" / "latency.csv"
    row1 = ingest_table(csv_path, CORPUS_DIR, FakeEmbeddingClient())
    row2 = ingest_table(csv_path, CORPUS_DIR, FakeEmbeddingClient())
    assert row1.id == row2.id
    assert row1.content_text == row2.content_text
    assert row1.vector == row2.vector
