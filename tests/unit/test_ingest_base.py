from pathlib import Path

import fitz
import pytest
from PIL import Image

from mmsearch import db
from mmsearch.clients.fakes import FakeCaptioner, FakeEmbeddingClient
from mmsearch.clients.protocols import Embedders
from mmsearch.ingest.base import IngestStats, classify_file, ingest_corpus, walk_corpus

EMBEDDERS = Embedders(image=FakeEmbeddingClient(), text=FakeEmbeddingClient())


# --- classify_file (dispatch logic) ------------------------------------------------------

@pytest.mark.parametrize(
    "filename,expected",
    [
        ("specs/rfc.pdf", "pdf"),
        ("docs/auth-flow.png", "diagram"),
        ("docs/photo.jpg", "diagram"),
        ("docs/photo.jpeg", "diagram"),
        ("docs/anim.gif", "diagram"),
        ("docs/scan.bmp", "diagram"),
        ("docs/modern.webp", "diagram"),
        ("data/latency.csv", "table"),
        ("src/ingest/base.py", "code"),
    ],
)
def test_classify_file_recognizes_supported_extensions(filename, expected):
    assert classify_file(Path(filename)) == expected


def test_classify_file_is_case_insensitive_on_extension():
    assert classify_file(Path("docs/AUTH-FLOW.PNG")) == "diagram"


@pytest.mark.parametrize("filename", ["README.md", "notes.txt", "data.xlsx", ".gitignore", "archive.zip"])
def test_classify_file_returns_none_for_unsupported_extensions(filename):
    assert classify_file(Path(filename)) is None


# --- walk_corpus ---------------------------------------------------------------------------

def test_walk_corpus_finds_files_in_nested_directories(tmp_path):
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs" / "a.pdf").write_bytes(b"x")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "b.csv").write_text("x")

    found = walk_corpus(tmp_path)

    relpaths = {p.relative_to(tmp_path).as_posix() for p in found}
    assert relpaths == {"specs/a.pdf", "data/b.csv"}


def test_walk_corpus_skips_hidden_files_and_directories(tmp_path):
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs" / "a.pdf").write_bytes(b"x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x")
    (tmp_path / ".DS_Store").write_text("x")

    found = walk_corpus(tmp_path)

    relpaths = {p.relative_to(tmp_path).as_posix() for p in found}
    assert relpaths == {"specs/a.pdf"}


def test_walk_corpus_returns_deterministic_sorted_order(tmp_path):
    (tmp_path / "z.py").write_text("x")
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "m.py").write_text("x")

    found = walk_corpus(tmp_path)

    assert [p.name for p in found] == ["a.py", "m.py", "z.py"]


def test_walk_corpus_empty_directory_returns_empty_list(tmp_path):
    assert walk_corpus(tmp_path) == []


# --- ingest_corpus (end-to-end driver) ------------------------------------------------------

def _make_mixed_corpus(root: Path) -> None:
    (root / "specs").mkdir()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello from a real text layer.")
    doc.save(root / "specs" / "doc.pdf")
    doc.close()

    (root / "docs").mkdir()
    img = Image.new("RGB", (10, 10), "white")
    img.save(root / "docs" / "diagram.png")

    (root / "data").mkdir()
    (root / "data" / "table.csv").write_text("a,b\n1,2\n")

    (root / "src").mkdir()
    (root / "src" / "code.py").write_text("def greet():\n    return 'hi'\n")

    (root / "README.md").write_text("not ingestable")


def test_ingest_corpus_writes_rows_for_every_supported_modality(tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    _make_mixed_corpus(corpus_root)
    table = db.open_table(uri=tmp_path / "lancedb")

    stats = ingest_corpus(
        corpus_root,
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
        thumbnails_dir=tmp_path / "thumbnails",
    )

    assert stats.rows_by_modality == {"pdf_page": 1, "diagram": 1, "table": 1, "code": 1}
    assert stats.rows_written == 4
    assert table.count_rows() == 4


def test_ingest_corpus_records_unsupported_files_as_skipped(tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    _make_mixed_corpus(corpus_root)
    table = db.open_table(uri=tmp_path / "lancedb")

    stats = ingest_corpus(
        corpus_root,
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
        thumbnails_dir=tmp_path / "thumbnails",
    )

    skipped_paths = {relpath for relpath, _reason in stats.skipped}
    assert "README.md" in skipped_paths


def test_ingest_corpus_is_idempotent_on_rerun(tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    _make_mixed_corpus(corpus_root)
    table = db.open_table(uri=tmp_path / "lancedb")

    ingest_corpus(
        corpus_root, EMBEDDERS, FakeCaptioner(), table, thumbnails_dir=tmp_path / "thumbnails"
    )
    ingest_corpus(
        corpus_root, EMBEDDERS, FakeCaptioner(), table, thumbnails_dir=tmp_path / "thumbnails"
    )

    assert table.count_rows() == 4  # upsert dedups by id, no duplicates


def test_ingest_corpus_continues_after_a_malformed_file(tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    _make_mixed_corpus(corpus_root)
    (corpus_root / "specs" / "broken.pdf").write_bytes(b"not a real pdf")
    table = db.open_table(uri=tmp_path / "lancedb")

    stats = ingest_corpus(
        corpus_root,
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
        thumbnails_dir=tmp_path / "thumbnails",
    )

    # the good pdf still gets ingested despite the broken one failing
    assert stats.rows_by_modality.get("pdf_page") == 1
    failed_paths = {relpath for relpath, _reason in stats.skipped}
    assert "specs/broken.pdf" in failed_paths


def test_ingest_stats_tracks_text_source_breakdown(tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    _make_mixed_corpus(corpus_root)
    table = db.open_table(uri=tmp_path / "lancedb")

    stats = ingest_corpus(
        corpus_root,
        embedders=EMBEDDERS,
        captioner=FakeCaptioner(),
        table=table,
        thumbnails_dir=tmp_path / "thumbnails",
    )

    assert stats.rows_by_text_source["pdf_text_layer"] == 1  # doc.pdf has a real text layer
    assert stats.rows_by_text_source["vlm_caption"] == 1  # diagram.png has no text layer
    assert stats.rows_by_text_source["table_markdown"] == 1
    assert stats.rows_by_text_source["code_source"] == 1


def test_ingest_stats_is_a_fresh_dataclass_instance():
    stats = IngestStats()
    assert stats.rows_written == 0
    assert stats.rows_by_modality == {}
    assert stats.skipped == []
