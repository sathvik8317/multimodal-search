import json

from mmsearch import config
from mmsearch.clients.fakes import FakeEmbeddingClient
from mmsearch.ingest.code import ingest_code_file
from tests.fixtures import CORPUS_DIR


def _rows():
    path = CORPUS_DIR / "src" / "ingest" / "base.py"
    return ingest_code_file(path, CORPUS_DIR, FakeEmbeddingClient())


def _row_by_id(rows, row_id):
    for row in rows:
        if row.id == row_id:
            return row
    raise AssertionError(f"no row with id {row_id!r} in {[r.id for r in rows]}")


# --- row count -------------------------------------------------------------------------

def test_ingest_code_file_returns_exactly_two_rows():
    rows = _rows()
    assert len(rows) == 2


# --- method row --------------------------------------------------------------------------

def test_method_row_has_expected_id():
    rows = _rows()
    ids = {r.id for r in rows}
    assert "code:src/ingest/base.py#PdfIngester.rasterize" in ids


def test_method_row_metadata():
    rows = _rows()
    row = _row_by_id(rows, "code:src/ingest/base.py#PdfIngester.rasterize")
    metadata = json.loads(row.metadata)
    assert metadata["kind"] == "method"
    assert metadata["lang"] == "python"
    assert metadata["qualname"] == "PdfIngester.rasterize"
    assert metadata["start_line"] == 7
    assert metadata["end_line"] == 9


def test_method_row_content_contains_signature_and_body():
    rows = _rows()
    row = _row_by_id(rows, "code:src/ingest/base.py#PdfIngester.rasterize")
    assert "rasterize" in row.content_text
    assert "get_pixmap" in row.content_text


# --- function row ------------------------------------------------------------------------

def test_function_row_has_expected_id():
    rows = _rows()
    ids = {r.id for r in rows}
    assert "code:src/ingest/base.py#_embed_and_write" in ids


def test_function_row_metadata():
    rows = _rows()
    row = _row_by_id(rows, "code:src/ingest/base.py#_embed_and_write")
    metadata = json.loads(row.metadata)
    assert metadata["kind"] == "function"
    assert metadata["lang"] == "python"
    assert metadata["qualname"] == "_embed_and_write"


def test_function_row_content_contains_name_and_docstring_text():
    rows = _rows()
    row = _row_by_id(rows, "code:src/ingest/base.py#_embed_and_write")
    assert "_embed_and_write" in row.content_text
    assert "exponential" in row.content_text


# --- start/end line sanity ----------------------------------------------------------------

def test_start_and_end_lines_are_internally_consistent_and_nonoverlapping():
    rows = _rows()
    method_row = _row_by_id(rows, "code:src/ingest/base.py#PdfIngester.rasterize")
    func_row = _row_by_id(rows, "code:src/ingest/base.py#_embed_and_write")
    method_meta = json.loads(method_row.metadata)
    func_meta = json.loads(func_row.metadata)

    assert method_meta["end_line"] >= method_meta["start_line"]
    assert func_meta["end_line"] >= func_meta["start_line"]

    # the two symbols must not overlap
    assert method_meta["end_line"] < func_meta["start_line"]


# --- vectors ---------------------------------------------------------------------------

def test_every_row_vector_has_correct_dim():
    rows = _rows()
    for row in rows:
        assert len(row.vector) == config.EMBED_DIM


# --- determinism -------------------------------------------------------------------------

def test_ingesting_same_file_twice_is_deterministic():
    rows_a = _rows()
    rows_b = _rows()
    ids_a = [r.id for r in rows_a]
    ids_b = [r.id for r in rows_b]
    assert ids_a == ids_b

    texts_a = [r.content_text for r in rows_a]
    texts_b = [r.content_text for r in rows_b]
    assert texts_a == texts_b


# --- source_path / text_source / thumbnail_ref -------------------------------------------

def test_row_source_path_and_text_source_and_thumbnail_ref():
    rows = _rows()
    for row in rows:
        assert row.source_path == "src/ingest/base.py"
        assert row.text_source.value == "code_source"
        assert row.thumbnail_ref == ""


# --- context header format -----------------------------------------------------------------

def test_context_header_includes_file_and_language():
    rows = _rows()
    row = _row_by_id(rows, "code:src/ingest/base.py#_embed_and_write")
    assert "# file: src/ingest/base.py" in row.content_text
    assert "# language: python" in row.content_text


def test_context_header_includes_class_for_method_but_not_function():
    rows = _rows()
    method_row = _row_by_id(rows, "code:src/ingest/base.py#PdfIngester.rasterize")
    func_row = _row_by_id(rows, "code:src/ingest/base.py#_embed_and_write")
    assert "# class: PdfIngester" in method_row.content_text
    assert "# class:" not in func_row.content_text
