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
    assert row.vector_cohere is None
    assert len(row.vector_openai) == config.OPENAI_EMBED_DIM


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


# --- row cap (MAX_TABLE_ROWS) -------------------------------------------------------------

def _write_csv(path, n_data_rows: int) -> None:
    lines = ["id,value"]
    lines += [f"{i},{i * 2}" for i in range(n_data_rows)]
    path.write_text("\n".join(lines) + "\n")


def test_ingest_table_under_cap_is_unaffected(tmp_path):
    csv_path = tmp_path / "small.csv"
    _write_csv(csv_path, 5)

    row = ingest_table(csv_path, tmp_path, FakeEmbeddingClient())

    metadata = json.loads(row.metadata)
    assert metadata["n_rows"] == 5
    assert metadata["total_rows"] == 5
    assert metadata["truncated"] is False

    data_lines = row.content_text.strip().splitlines()[2:]  # skip header + separator
    assert len(data_lines) == 5


def test_ingest_table_over_cap_is_truncated(tmp_path):
    csv_path = tmp_path / "big.csv"
    _write_csv(csv_path, 250)  # over config.MAX_TABLE_ROWS (200)

    row = ingest_table(csv_path, tmp_path, FakeEmbeddingClient())

    metadata = json.loads(row.metadata)
    assert metadata["truncated"] is True
    assert metadata["total_rows"] == 250
    assert metadata["n_rows"] == config.MAX_TABLE_ROWS

    data_lines = row.content_text.strip().splitlines()[2:]  # skip header + separator
    assert len(data_lines) == config.MAX_TABLE_ROWS


def test_ingest_table_truncation_keeps_the_first_rows(tmp_path):
    csv_path = tmp_path / "big.csv"
    _write_csv(csv_path, 250)

    row = ingest_table(csv_path, tmp_path, FakeEmbeddingClient())

    assert "| 0 | 0 |" in row.content_text  # first data row present
    assert "| 199 | 398 |" in row.content_text  # 200th data row (index 199) present
    assert "| 200 | 400 |" not in row.content_text  # 201st data row absent


# --- char budget (MAX_TABLE_EMBED_CHARS) ---------------------------------------------------
#
# A wide table (many columns) can blow OpenAI's 8192-token input limit well
# before hitting MAX_TABLE_ROWS -- this is the real bug the migration
# introduced (see EMBEDDING_MIGRATION_PLAN.md): a 200-row cap sized for
# Cohere's much larger context silently produced embed inputs OpenAI
# rejected for 3 of 4 real corpus CSVs. These tests build a table wide
# enough that the char budget, not the row cap, is what binds.

def _write_wide_csv(path, n_data_rows: int, n_cols: int = 20, cell_width: int = 15) -> None:
    columns = [f"col{i}" for i in range(n_cols)]
    lines = [",".join(columns)]
    for row_i in range(n_data_rows):
        cell = "x" * cell_width
        lines.append(",".join(f"{cell}{row_i}" for _ in range(n_cols)))
    path.write_text("\n".join(lines) + "\n")


def test_ingest_table_wide_csv_is_truncated_by_char_budget_before_row_cap(tmp_path):
    csv_path = tmp_path / "wide.csv"
    _write_wide_csv(csv_path, n_data_rows=150, n_cols=20, cell_width=15)  # well under MAX_TABLE_ROWS

    row = ingest_table(csv_path, tmp_path, FakeEmbeddingClient())

    metadata = json.loads(row.metadata)
    assert metadata["truncated"] is True
    assert metadata["n_rows"] < 150  # char budget bound before the row cap did
    assert len(row.content_text) <= config.MAX_TABLE_EMBED_CHARS


def test_ingest_table_char_budget_always_keeps_at_least_one_row(tmp_path):
    csv_path = tmp_path / "huge_row.csv"
    # A single row alone exceeds MAX_TABLE_EMBED_CHARS -- must still be kept,
    # not dropped to zero rows (which would also violate content_text
    # must-not-be-empty).
    _write_wide_csv(csv_path, n_data_rows=1, n_cols=20, cell_width=1000)

    row = ingest_table(csv_path, tmp_path, FakeEmbeddingClient())

    metadata = json.loads(row.metadata)
    assert metadata["n_rows"] == 1
    assert metadata["truncated"] is False  # the only row available was kept, nothing was dropped
    data_lines = row.content_text.strip().splitlines()[2:]
    assert len(data_lines) == 1


def test_ingest_table_narrow_csv_under_row_cap_is_unaffected_by_char_budget(tmp_path):
    # Sanity check: the existing narrow-CSV row-cap tests above must still
    # bind on MAX_TABLE_ROWS, not on the new char budget.
    csv_path = tmp_path / "narrow_big.csv"
    _write_csv(csv_path, 250)

    row = ingest_table(csv_path, tmp_path, FakeEmbeddingClient())

    metadata = json.loads(row.metadata)
    assert metadata["n_rows"] == config.MAX_TABLE_ROWS
    assert len(row.content_text) <= config.MAX_TABLE_EMBED_CHARS


def test_ingest_table_is_deterministic():
    csv_path = CORPUS_DIR / "data" / "latency.csv"
    row1 = ingest_table(csv_path, CORPUS_DIR, FakeEmbeddingClient())
    row2 = ingest_table(csv_path, CORPUS_DIR, FakeEmbeddingClient())
    assert row1.id == row2.id
    assert row1.content_text == row2.content_text
    assert row1.vector_openai == row2.vector_openai
