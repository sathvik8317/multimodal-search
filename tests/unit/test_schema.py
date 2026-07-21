import pyarrow as pa
import pytest

from mmsearch import config
from mmsearch.schema import ARROW_SCHEMA, Modality, Row, TextSource, make_id, rows_to_table


# --- Modality / TextSource enums ---------------------------------------------------

def test_modality_values():
    assert Modality.PDF_PAGE.value == "pdf_page"
    assert Modality.DIAGRAM.value == "diagram"
    assert Modality.TABLE.value == "table"
    assert Modality.CODE.value == "code"


def test_text_source_values():
    assert TextSource.PDF_TEXT_LAYER.value == "pdf_text_layer"
    assert TextSource.VLM_CAPTION.value == "vlm_caption"
    assert TextSource.TABLE_MARKDOWN.value == "table_markdown"
    assert TextSource.CODE_SOURCE.value == "code_source"


# --- make_id ------------------------------------------------------------------------

def test_make_id_pdf_page():
    assert make_id(Modality.PDF_PAGE, "specs/rfc.pdf", page_no=14) == "pdf:specs/rfc.pdf#p14"


def test_make_id_diagram():
    assert make_id(Modality.DIAGRAM, "docs/auth-flow.png") == "img:docs/auth-flow.png"


def test_make_id_table():
    assert make_id(Modality.TABLE, "data/latency.csv") == "tbl:data/latency.csv"


def test_make_id_code():
    assert (
        make_id(Modality.CODE, "src/ingest/base.py", qualname="PdfIngester.rasterize")
        == "code:src/ingest/base.py#PdfIngester.rasterize"
    )


def test_make_id_normalizes_windows_separators():
    # relpath must always come out POSIX-separated, stable across OSes
    assert make_id(Modality.DIAGRAM, "docs\\auth-flow.png") == "img:docs/auth-flow.png"


def test_make_id_strips_leading_slash():
    assert make_id(Modality.TABLE, "/data/latency.csv") == "tbl:data/latency.csv"


def test_make_id_pdf_page_requires_page_no():
    with pytest.raises(ValueError, match="page_no"):
        make_id(Modality.PDF_PAGE, "specs/rfc.pdf")


def test_make_id_code_requires_qualname():
    with pytest.raises(ValueError, match="qualname"):
        make_id(Modality.CODE, "src/ingest/base.py")


def test_make_id_is_deterministic():
    a = make_id(Modality.PDF_PAGE, "specs/rfc.pdf", page_no=14)
    b = make_id(Modality.PDF_PAGE, "specs/rfc.pdf", page_no=14)
    assert a == b


# --- Row dataclass --------------------------------------------------------------------

def test_row_defaults():
    row = Row(
        id="tbl:data/latency.csv",
        modality=Modality.TABLE,
        content_text="| a | b |\n|---|---|\n| 1 | 2 |",
        text_source=TextSource.TABLE_MARKDOWN,
        vector_openai=[0.0] * config.OPENAI_EMBED_DIM,
        source_path="data/latency.csv",
    )
    assert row.thumbnail_ref == ""
    assert row.metadata == "{}"


def test_row_requires_at_least_one_vector():
    with pytest.raises(ValueError, match="at least one"):
        Row(
            id="tbl:data/latency.csv",
            modality=Modality.TABLE,
            content_text="text",
            text_source=TextSource.TABLE_MARKDOWN,
            source_path="data/latency.csv",
        )


def test_row_vector_cohere_must_match_embed_dim():
    with pytest.raises(ValueError, match="COHERE_EMBED_DIM"):
        Row(
            id="tbl:data/latency.csv",
            modality=Modality.TABLE,
            content_text="text",
            text_source=TextSource.TABLE_MARKDOWN,
            vector_cohere=[0.0, 0.0],  # wrong length
            source_path="data/latency.csv",
        )


def test_row_vector_openai_must_match_embed_dim():
    with pytest.raises(ValueError, match="OPENAI_EMBED_DIM"):
        Row(
            id="tbl:data/latency.csv",
            modality=Modality.TABLE,
            content_text="text",
            text_source=TextSource.TABLE_MARKDOWN,
            vector_openai=[0.0, 0.0],  # wrong length
            source_path="data/latency.csv",
        )


def test_row_content_text_must_not_be_empty():
    with pytest.raises(ValueError, match="content_text"):
        Row(
            id="tbl:data/latency.csv",
            modality=Modality.TABLE,
            content_text="",
            text_source=TextSource.TABLE_MARKDOWN,
            vector_openai=[0.0] * config.OPENAI_EMBED_DIM,
            source_path="data/latency.csv",
        )


# --- Arrow schema ---------------------------------------------------------------------

def test_arrow_schema_field_names():
    assert ARROW_SCHEMA.names == [
        "id",
        "modality",
        "content_text",
        "text_source",
        "vector_cohere",
        "vector_openai",
        "source_path",
        "thumbnail_ref",
        "metadata",
    ]


def test_arrow_schema_vector_cohere_is_fixed_size_list_float32():
    vector_field = ARROW_SCHEMA.field("vector_cohere")
    assert pa.types.is_fixed_size_list(vector_field.type)
    assert vector_field.type.list_size == config.COHERE_EMBED_DIM
    assert vector_field.type.value_type == pa.float32()


def test_arrow_schema_vector_openai_is_fixed_size_list_float32():
    vector_field = ARROW_SCHEMA.field("vector_openai")
    assert pa.types.is_fixed_size_list(vector_field.type)
    assert vector_field.type.list_size == config.OPENAI_EMBED_DIM
    assert vector_field.type.value_type == pa.float32()


def test_arrow_schema_string_columns():
    for name in ("id", "modality", "content_text", "text_source", "source_path", "thumbnail_ref", "metadata"):
        assert ARROW_SCHEMA.field(name).type == pa.string()


# --- rows_to_table ----------------------------------------------------------------------

def test_rows_to_table_roundtrips_single_row():
    row = Row(
        id="code:src/a.py#f",
        modality=Modality.CODE,
        content_text="def f(): pass",
        text_source=TextSource.CODE_SOURCE,
        vector_openai=[0.5] * config.OPENAI_EMBED_DIM,
        source_path="src/a.py",
        metadata='{"lang": "python"}',
    )
    table = rows_to_table([row])
    assert table.schema == ARROW_SCHEMA
    assert table.num_rows == 1
    assert table.column("id").to_pylist() == ["code:src/a.py#f"]
    assert table.column("modality").to_pylist() == ["code"]
    assert table.column("text_source").to_pylist() == ["code_source"]
    assert table.column("vector_cohere").to_pylist() == [None]
    assert table.column("metadata").to_pylist() == ['{"lang": "python"}']


def test_rows_to_table_empty_list_has_zero_rows_and_correct_schema():
    table = rows_to_table([])
    assert table.schema == ARROW_SCHEMA
    assert table.num_rows == 0
