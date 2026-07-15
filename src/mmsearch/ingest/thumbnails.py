"""Thumbnail writer: persists PNG bytes under a thumbnails directory."""

from __future__ import annotations

from pathlib import Path

from mmsearch import config


def write_thumbnail(
    image_bytes: bytes,
    relpath: str,
    thumbnails_dir: Path = config.THUMBNAILS_DIR,
) -> str:
    """Write ``image_bytes`` as a PNG under ``thumbnails_dir`` at ``relpath``.

    Returns the path relative to ``thumbnails_dir`` (POSIX-separated), suitable
    for ``Row.thumbnail_ref``.
    """
    normalized = relpath.replace("\\", "/").lstrip("/")
    dest = thumbnails_dir / normalized
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(image_bytes)
    return normalized
