from pathlib import Path

from PIL import Image

from mmsearch.ingest.thumbnails import write_thumbnail


def _make_png_bytes(width: int = 10, height: int = 10) -> bytes:
    import io

    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_write_thumbnail_writes_a_real_openable_png(tmp_path: Path):
    thumbnails_dir = tmp_path / "thumbs"
    png_bytes = _make_png_bytes()

    relref = write_thumbnail(png_bytes, "docs/auth-flow.png", thumbnails_dir=thumbnails_dir)

    full_path = thumbnails_dir / relref
    assert full_path.exists()
    with Image.open(full_path) as img:
        assert img.format == "PNG"


def test_write_thumbnail_returns_path_relative_to_thumbnails_dir(tmp_path: Path):
    thumbnails_dir = tmp_path / "thumbs"
    png_bytes = _make_png_bytes()

    relref = write_thumbnail(png_bytes, "docs/auth-flow.png", thumbnails_dir=thumbnails_dir)

    assert relref == "docs/auth-flow.png"
    assert not Path(relref).is_absolute()


def test_write_thumbnail_creates_parent_directories_for_nested_relpath(tmp_path: Path):
    thumbnails_dir = tmp_path / "thumbs"
    png_bytes = _make_png_bytes()

    relref = write_thumbnail(png_bytes, "a/b/c/deep.png", thumbnails_dir=thumbnails_dir)

    full_path = thumbnails_dir / relref
    assert full_path.parent.is_dir()
    assert full_path.exists()


def test_write_thumbnail_relpath_is_posix_separated(tmp_path: Path):
    thumbnails_dir = tmp_path / "thumbs"
    png_bytes = _make_png_bytes()

    relref = write_thumbnail(png_bytes, "a/b/c/deep.png", thumbnails_dir=thumbnails_dir)

    assert "\\" not in relref
