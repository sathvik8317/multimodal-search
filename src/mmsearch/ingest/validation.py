"""Content-vs-extension validation for uploaded files.

Uploads are untrusted -- unlike ingest_corpus's directory walk, extension
alone can't be trusted here (a renamed .exe with a .pdf name would otherwise
sail straight into fitz.open()). Binary types are checked by magic-byte
signature; the two text types (.py/.csv) have no reliable magic bytes, so
they're checked for decodability instead.
"""

from __future__ import annotations

_MAGIC_SIGNATURES: dict[str, bytes] = {
    ".pdf": b"%PDF-",
    ".png": b"\x89PNG\r\n\x1a\n",
    ".jpg": b"\xff\xd8\xff",
    ".jpeg": b"\xff\xd8\xff",
    ".gif": b"GIF8",
    ".bmp": b"BM",
    ".xlsx": b"PK\x03\x04",  # xlsx is a zip archive
}

_TEXT_SUFFIXES = {".py", ".csv"}

_SNIFF_SAMPLE_BYTES = 8192


class UploadValidationError(ValueError):
    pass


def _validate_webp(data: bytes) -> None:
    if not (data[:4] == b"RIFF" and data[8:12] == b"WEBP"):
        raise UploadValidationError("file content does not match .webp magic bytes")


def _validate_text(suffix: str, data: bytes) -> None:
    try:
        sample = data[:_SNIFF_SAMPLE_BYTES].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UploadValidationError(f"{suffix} file is not valid UTF-8 text") from exc
    if "\x00" in sample:
        raise UploadValidationError(f"{suffix} file contains NUL bytes, not real text")


def validate_upload_content(suffix: str, data: bytes) -> None:
    """Raise UploadValidationError if data's content doesn't match suffix."""
    suffix = suffix.lower()

    if suffix in _TEXT_SUFFIXES:
        _validate_text(suffix, data)
        return

    if suffix == ".webp":
        _validate_webp(data)
        return

    signature = _MAGIC_SIGNATURES.get(suffix)
    if signature is None:
        raise UploadValidationError(f"unsupported extension {suffix!r}")
    if not data.startswith(signature):
        raise UploadValidationError(f"file content does not match {suffix} magic bytes")
