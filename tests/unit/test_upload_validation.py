"""Tests for magic-byte content sniffing on uploaded files -- the trust
boundary check that stops a renamed/mislabeled file from reaching ingest just
because its extension matched the allowlist."""

import pytest

from mmsearch.ingest.validation import UploadValidationError, validate_upload_content

_REAL_PDF = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n..."
_REAL_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
_REAL_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 20
_REAL_GIF = b"GIF89a" + b"\x00" * 20
_REAL_BMP = b"BM" + b"\x00" * 20
_REAL_WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 20
_REAL_XLSX = b"PK\x03\x04" + b"\x00" * 20  # xlsx is a zip archive
_NOT_A_REAL_FILE = b"MZ\x90\x00\x03\x00\x00\x00"  # a Windows .exe (PE) header


# --- valid content is accepted for its matching extension ----------------------------------

@pytest.mark.parametrize(
    "suffix,data",
    [
        (".pdf", _REAL_PDF),
        (".png", _REAL_PNG),
        (".jpg", _REAL_JPEG),
        (".jpeg", _REAL_JPEG),
        (".gif", _REAL_GIF),
        (".bmp", _REAL_BMP),
        (".webp", _REAL_WEBP),
        (".xlsx", _REAL_XLSX),
    ],
)
def test_accepts_content_matching_its_extension(suffix, data):
    validate_upload_content(suffix, data)  # must not raise


def test_accepts_extension_case_insensitively():
    validate_upload_content(".PDF", _REAL_PDF)  # must not raise


# --- mismatched/renamed binary content is rejected ------------------------------------------

@pytest.mark.parametrize(
    "suffix", [".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".xlsx"]
)
def test_rejects_renamed_exe_for_every_binary_extension(suffix):
    with pytest.raises(UploadValidationError):
        validate_upload_content(suffix, _NOT_A_REAL_FILE)


def test_webp_requires_both_riff_and_webp_markers():
    # RIFF alone (e.g. a renamed .wav/.avi, also RIFF-based) must not pass as webp.
    riff_but_not_webp = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 20
    with pytest.raises(UploadValidationError):
        validate_upload_content(".webp", riff_but_not_webp)


# --- text types: validated by decodability, not magic bytes --------------------------------

def test_py_accepts_valid_utf8_text():
    validate_upload_content(".py", "def greet():\n    return 'hi'\n".encode("utf-8"))


def test_csv_accepts_valid_utf8_text():
    validate_upload_content(".csv", "a,b\n1,2\n".encode("utf-8"))


def test_py_rejects_invalid_utf8():
    with pytest.raises(UploadValidationError):
        validate_upload_content(".py", b"\xff\xfe\x00\x01invalid utf-8")


def test_py_rejects_content_with_nul_bytes():
    with pytest.raises(UploadValidationError):
        validate_upload_content(".py", b"looks like text\x00but has a NUL byte")


# --- unsupported extension -------------------------------------------------------------------

def test_unsupported_extension_raises():
    with pytest.raises(UploadValidationError):
        validate_upload_content(".exe", _NOT_A_REAL_FILE)
