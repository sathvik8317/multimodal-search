"""Tests for the single-file upload orchestration (classify -> validate ->
ingest -> stamp provenance -> upsert). HTTP concerns (multipart parsing, size
cap, rate limiting) are not this module's job -- see api/main.py's /upload
route, which is a thin wrapper around ingest_uploaded_file."""

import io
import json

import fitz
import pytest
from PIL import Image

from mmsearch import db
from mmsearch.clients.fakes import FakeCaptioner, FakeEmbeddingClient
from mmsearch.clients.protocols import Embedders
from mmsearch.ingest.upload import (
    UnsupportedUploadError,
    ingest_uploaded_file,
)
from mmsearch.ingest.validation import UploadValidationError
from mmsearch.schema import Modality

EMBEDDERS = Embedders(image=FakeEmbeddingClient(), text=FakeEmbeddingClient())


def _real_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buf, format="PNG")
    return buf.getvalue()


def _real_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello from a real text layer.")
    data = doc.tobytes()
    doc.close()
    return data


def _table(tmp_path):
    return db.open_table(uri=tmp_path / "lancedb")


# --- code upload -----------------------------------------------------------------------

def test_ingest_uploaded_code_file_writes_rows(tmp_path):
    table = _table(tmp_path)

    result = ingest_uploaded_file(
        filename="greet.py",
        data=b"def greet():\n    return 'hi'\n",
        uploader="alice",
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
        staging_root=tmp_path / "staging",
        thumbnails_dir=tmp_path / "thumbnails",
    )

    assert result.modality == "code"
    assert result.rows_written == 1
    assert table.count_rows() == 1


def test_ingest_uploaded_code_file_source_path_carries_uploads_prefix(tmp_path):
    table = _table(tmp_path)

    ingest_uploaded_file(
        filename="greet.py",
        data=b"def greet():\n    return 'hi'\n",
        uploader="alice",
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
        staging_root=tmp_path / "staging",
        thumbnails_dir=tmp_path / "thumbnails",
    )

    row = table.to_arrow().to_pylist()[0]
    assert row["source_path"].startswith("uploads/alice/")


def test_ingest_uploaded_code_file_stamps_uploader_and_uploaded_at(tmp_path):
    table = _table(tmp_path)

    ingest_uploaded_file(
        filename="greet.py",
        data=b"def greet():\n    return 'hi'\n",
        uploader="alice",
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
        staging_root=tmp_path / "staging",
        thumbnails_dir=tmp_path / "thumbnails",
    )

    row = table.to_arrow().to_pylist()[0]
    metadata = json.loads(row["metadata"])
    assert metadata["uploader"] == "alice"
    assert "uploaded_at" in metadata and metadata["uploaded_at"]


# --- table upload (csv) -----------------------------------------------------------------

def test_ingest_uploaded_csv_file_writes_a_table_row(tmp_path):
    table = _table(tmp_path)

    result = ingest_uploaded_file(
        filename="data.csv",
        data=b"a,b\n1,2\n",
        uploader="bob",
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
        staging_root=tmp_path / "staging",
        thumbnails_dir=tmp_path / "thumbnails",
    )

    assert result.modality == "table"
    assert result.rows_written == 1


# --- diagram upload (png) ---------------------------------------------------------------

def test_ingest_uploaded_png_writes_a_diagram_row_and_thumbnail(tmp_path):
    table = _table(tmp_path)
    thumbnails_dir = tmp_path / "thumbnails"

    result = ingest_uploaded_file(
        filename="diagram.png",
        data=_real_png_bytes(),
        uploader="carol",
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
        staging_root=tmp_path / "staging",
        thumbnails_dir=thumbnails_dir,
    )

    assert result.modality == "diagram"
    assert result.rows_written == 1
    assert len(result.thumbnail_refs) == 1
    assert (thumbnails_dir / result.thumbnail_refs[0]).is_file()


# --- pdf upload --------------------------------------------------------------------------

def test_ingest_uploaded_pdf_writes_a_pdf_page_row(tmp_path):
    table = _table(tmp_path)

    result = ingest_uploaded_file(
        filename="paper.pdf",
        data=_real_pdf_bytes(),
        uploader="dave",
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
        staging_root=tmp_path / "staging",
        thumbnails_dir=tmp_path / "thumbnails",
    )

    assert result.modality == "pdf_page"
    assert result.rows_written == 1


# --- uploader sanitization/default -------------------------------------------------------

def test_uploader_none_defaults_to_anon(tmp_path):
    table = _table(tmp_path)

    ingest_uploaded_file(
        filename="greet.py",
        data=b"def greet():\n    return 'hi'\n",
        uploader=None,
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
        staging_root=tmp_path / "staging",
        thumbnails_dir=tmp_path / "thumbnails",
    )

    row = table.to_arrow().to_pylist()[0]
    assert row["source_path"].startswith("uploads/anon/")


def test_uploader_with_unsafe_characters_is_sanitized(tmp_path):
    table = _table(tmp_path)

    ingest_uploaded_file(
        filename="greet.py",
        data=b"def greet():\n    return 'hi'\n",
        uploader="../../etc/passwd; rm -rf",
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
        staging_root=tmp_path / "staging",
        thumbnails_dir=tmp_path / "thumbnails",
    )

    row = table.to_arrow().to_pylist()[0]
    uploader_dir = row["source_path"].split("/")[1]
    assert "/" not in uploader_dir
    assert ".." not in uploader_dir


# --- filename sanitization (path traversal) -----------------------------------------------

def test_filename_with_path_traversal_is_sanitized_to_basename(tmp_path):
    table = _table(tmp_path)
    staging_root = tmp_path / "staging"

    ingest_uploaded_file(
        filename="../../../etc/passwd.py",
        data=b"def f(): pass\n",
        uploader="alice",
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
        staging_root=staging_root,
        thumbnails_dir=tmp_path / "thumbnails",
    )

    # nothing was written outside staging_root
    written_files = list(staging_root.rglob("*.py"))
    assert len(written_files) == 1
    assert written_files[0].is_relative_to(staging_root)
    assert written_files[0].name == "passwd.py"


# --- unsupported extension -----------------------------------------------------------------

def test_unsupported_extension_raises_unsupported_upload_error(tmp_path):
    table = _table(tmp_path)

    with pytest.raises(UnsupportedUploadError):
        ingest_uploaded_file(
            filename="notes.txt",
            data=b"just some notes",
            uploader="alice",
            embedders=EMBEDDERS,
            captioner=FakeCaptioner(),
            table=table,
            staging_root=tmp_path / "staging",
            thumbnails_dir=tmp_path / "thumbnails",
        )
    assert table.count_rows() == 0


# --- content/extension mismatch -------------------------------------------------------------

def test_content_extension_mismatch_raises_upload_validation_error(tmp_path):
    table = _table(tmp_path)

    with pytest.raises(UploadValidationError):
        ingest_uploaded_file(
            filename="fake.pdf",
            data=b"this is not a real pdf, just text pretending to be one",
            uploader="alice",
            embedders=EMBEDDERS,
            captioner=FakeCaptioner(),
            table=table,
            staging_root=tmp_path / "staging",
            thumbnails_dir=tmp_path / "thumbnails",
        )
    assert table.count_rows() == 0


def test_validation_failure_writes_nothing_to_staging_root(tmp_path):
    table = _table(tmp_path)
    staging_root = tmp_path / "staging"

    with pytest.raises(UploadValidationError):
        ingest_uploaded_file(
            filename="fake.pdf",
            data=b"not a real pdf",
            uploader="alice",
            embedders=EMBEDDERS,
            captioner=FakeCaptioner(),
            table=table,
            staging_root=staging_root,
            thumbnails_dir=tmp_path / "thumbnails",
        )

    assert not staging_root.exists() or list(staging_root.rglob("*.pdf")) == []
