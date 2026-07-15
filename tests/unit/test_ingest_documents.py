import json
from pathlib import Path

import fitz
import pytest

from mmsearch import config
from mmsearch.clients.fakes import FakeCaptioner, FakeEmbeddingClient
from mmsearch.ingest.documents import ingest_diagram, ingest_pdf
from mmsearch.schema import Modality, TextSource
from tests.fixtures import CORPUS_DIR


# --- ingest_pdf: real fixture PDF (has a real text layer) ---------------------------

def test_ingest_pdf_returns_one_row_per_page(tmp_path: Path):
    pdf_path = CORPUS_DIR / "specs" / "rfc.pdf"
    rows = ingest_pdf(
        pdf_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs",
    )
    assert len(rows) == 2


def test_ingest_pdf_row_ids_are_correct(tmp_path: Path):
    pdf_path = CORPUS_DIR / "specs" / "rfc.pdf"
    rows = ingest_pdf(
        pdf_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs",
    )
    assert rows[0].id == "pdf:specs/rfc.pdf#p1"
    assert rows[1].id == "pdf:specs/rfc.pdf#p2"


def test_ingest_pdf_uses_text_layer_when_present(tmp_path: Path):
    pdf_path = CORPUS_DIR / "specs" / "rfc.pdf"
    rows = ingest_pdf(
        pdf_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs",
    )
    assert rows[0].text_source == TextSource.PDF_TEXT_LAYER
    assert rows[1].text_source == TextSource.PDF_TEXT_LAYER


def test_ingest_pdf_content_text_contains_expected_substrings(tmp_path: Path):
    pdf_path = CORPUS_DIR / "specs" / "rfc.pdf"
    rows = ingest_pdf(
        pdf_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs",
    )
    assert "retry backoff" in rows[0].content_text
    assert "p99 latency" in rows[1].content_text


def test_ingest_pdf_vector_has_correct_dimension(tmp_path: Path):
    pdf_path = CORPUS_DIR / "specs" / "rfc.pdf"
    rows = ingest_pdf(
        pdf_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs",
    )
    for row in rows:
        assert len(row.vector) == config.EMBED_DIM


def test_ingest_pdf_writes_real_thumbnail_files(tmp_path: Path):
    pdf_path = CORPUS_DIR / "specs" / "rfc.pdf"
    thumbnails_dir = tmp_path / "thumbs"
    rows = ingest_pdf(
        pdf_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=thumbnails_dir,
    )
    for row in rows:
        assert row.thumbnail_ref
        assert (thumbnails_dir / row.thumbnail_ref).exists()


def test_ingest_pdf_metadata_has_page_no_and_n_pages(tmp_path: Path):
    pdf_path = CORPUS_DIR / "specs" / "rfc.pdf"
    rows = ingest_pdf(
        pdf_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs",
    )
    meta0 = json.loads(rows[0].metadata)
    meta1 = json.loads(rows[1].metadata)
    assert meta0["page_no"] == 1
    assert meta0["n_pages"] == 2
    assert meta1["page_no"] == 2
    assert meta1["n_pages"] == 2


def test_ingest_pdf_source_path_is_corpus_relative(tmp_path: Path):
    pdf_path = CORPUS_DIR / "specs" / "rfc.pdf"
    rows = ingest_pdf(
        pdf_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs",
    )
    for row in rows:
        assert row.source_path == "specs/rfc.pdf"


def test_ingest_pdf_is_deterministic_across_runs(tmp_path: Path):
    pdf_path = CORPUS_DIR / "specs" / "rfc.pdf"
    rows1 = ingest_pdf(
        pdf_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs1",
    )
    rows2 = ingest_pdf(
        pdf_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs2",
    )
    assert [r.id for r in rows1] == [r.id for r in rows2]
    assert [r.vector for r in rows1] == [r.vector for r in rows2]


# --- ingest_pdf: synthetic textless (scanned) page falls back to captioning ---------

def _make_blank_pdf(path: Path) -> None:
    doc = fitz.open()
    try:
        doc.new_page(width=200, height=200)  # no insert_text call -> no text layer
        doc.save(path)
    finally:
        doc.close()


def test_ingest_pdf_falls_back_to_caption_for_textless_page(tmp_path: Path):
    pdf_path = tmp_path / "scanned.pdf"
    _make_blank_pdf(pdf_path)

    rows = ingest_pdf(
        pdf_path,
        tmp_path,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs",
    )

    assert len(rows) == 1
    assert rows[0].text_source == TextSource.VLM_CAPTION
    assert rows[0].content_text != ""


# --- ingest_diagram -------------------------------------------------------------------

def test_ingest_diagram_returns_row_with_correct_id(tmp_path: Path):
    image_path = CORPUS_DIR / "docs" / "auth-flow.png"
    row = ingest_diagram(
        image_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs",
    )
    assert row.id == "img:docs/auth-flow.png"


def test_ingest_diagram_always_uses_vlm_caption(tmp_path: Path):
    image_path = CORPUS_DIR / "docs" / "auth-flow.png"
    row = ingest_diagram(
        image_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs",
    )
    assert row.text_source == TextSource.VLM_CAPTION
    assert row.modality == Modality.DIAGRAM
    assert row.content_text != ""


def test_ingest_diagram_writes_thumbnail(tmp_path: Path):
    image_path = CORPUS_DIR / "docs" / "auth-flow.png"
    thumbnails_dir = tmp_path / "thumbs"
    row = ingest_diagram(
        image_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=thumbnails_dir,
    )
    assert row.thumbnail_ref
    assert (thumbnails_dir / row.thumbnail_ref).exists()


def test_ingest_diagram_metadata_has_width_height_and_caption_model(tmp_path: Path):
    image_path = CORPUS_DIR / "docs" / "auth-flow.png"
    row = ingest_diagram(
        image_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs",
    )
    metadata = json.loads(row.metadata)
    assert metadata["width"] == 600
    assert metadata["height"] == 400
    assert isinstance(metadata["caption_model"], str)
    assert metadata["caption_model"]


def test_ingest_diagram_vector_has_correct_dimension(tmp_path: Path):
    image_path = CORPUS_DIR / "docs" / "auth-flow.png"
    row = ingest_diagram(
        image_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs",
    )
    assert len(row.vector) == config.EMBED_DIM


def test_ingest_diagram_is_deterministic(tmp_path: Path):
    image_path = CORPUS_DIR / "docs" / "auth-flow.png"
    row1 = ingest_diagram(
        image_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs1",
    )
    row2 = ingest_diagram(
        image_path,
        CORPUS_DIR,
        FakeEmbeddingClient(),
        FakeCaptioner(),
        thumbnails_dir=tmp_path / "thumbs2",
    )
    assert row1.id == row2.id
    assert row1.content_text == row2.content_text
    assert row1.vector == row2.vector
