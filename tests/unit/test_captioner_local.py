"""Tests for the disk-cache wrapper and pure composition logic in
LocalCaptioner. The real model is never loaded here -- _query_fn is always
injected -- so these tests never import torch/transformers.
"""

import sys
from pathlib import Path

from mmsearch.clients.captioner_local import (
    _DEFAULT_MODEL_REVISION,
    LocalCaptioner,
    _compose_caption,
    _from_pretrained_kwargs,
)


def test_torch_is_not_imported_by_this_module():
    assert "torch" not in sys.modules


# --- model revision pinning ------------------------------------------------------------

def test_default_model_revision_is_pinned_not_none():
    captioner = LocalCaptioner(cache_dir=Path("unused"), _query_fn=lambda b: "x")
    assert captioner._model_revision == "2025-06-21"
    assert captioner._model_revision == _DEFAULT_MODEL_REVISION


def test_from_pretrained_kwargs_passes_through_the_pinned_revision():
    kwargs = _from_pretrained_kwargs(model_revision="2025-06-21", dtype="fp16-placeholder")
    assert kwargs["revision"] == "2025-06-21"


def test_from_pretrained_kwargs_always_includes_trust_remote_code_and_dtype():
    kwargs = _from_pretrained_kwargs(model_revision="2025-06-21", dtype="fp16-placeholder")
    assert kwargs["trust_remote_code"] is True
    assert kwargs["dtype"] == "fp16-placeholder"


def test_from_pretrained_kwargs_omits_revision_when_none():
    kwargs = _from_pretrained_kwargs(model_revision=None, dtype="fp16-placeholder")
    assert "revision" not in kwargs


def test_model_revision_is_configurable_via_constructor():
    captioner = LocalCaptioner(cache_dir=Path("unused"), model_revision="some-other-tag", _query_fn=lambda b: "x")
    assert captioner._model_revision == "some-other-tag"


# --- _compose_caption (pure) ---------------------------------------------------------

def test_compose_caption_includes_description_and_transcribed_text():
    result = _compose_caption("A diagram showing an auth flow.", "Client, Auth Server")
    assert "A diagram showing an auth flow." in result
    assert "Client, Auth Server" in result


def test_compose_caption_omits_transcription_section_when_no_text_visible():
    result = _compose_caption("A plain gradient background.", "")
    assert result == "A plain gradient background."


def test_compose_caption_omits_transcription_section_when_whitespace_only():
    result = _compose_caption("A blank page.", "   \n  ")
    assert result == "A blank page."


# --- disk cache wrapper (query_fn injected, no torch) ----------------------------------

def test_caption_calls_query_fn_and_returns_its_result(tmp_path):
    def fake_query(image_bytes: bytes) -> str:
        return "a fake caption"

    captioner = LocalCaptioner(cache_dir=tmp_path, _query_fn=fake_query)
    assert captioner.caption(b"some image bytes") == "a fake caption"


def test_caption_is_cached_on_second_call_with_same_bytes(tmp_path):
    calls = []

    def fake_query(image_bytes: bytes) -> str:
        calls.append(image_bytes)
        return "caption text"

    captioner = LocalCaptioner(cache_dir=tmp_path, _query_fn=fake_query)
    captioner.caption(b"same bytes")
    captioner.caption(b"same bytes")

    assert len(calls) == 1  # second call was served from cache


def test_caption_cache_is_keyed_by_image_content_not_call_order(tmp_path):
    calls = []

    def fake_query(image_bytes: bytes) -> str:
        calls.append(image_bytes)
        return f"caption for {len(image_bytes)} bytes"

    captioner = LocalCaptioner(cache_dir=tmp_path, _query_fn=fake_query)
    result_a = captioner.caption(b"image one")
    result_b = captioner.caption(b"image two, longer")

    assert len(calls) == 2
    assert result_a != result_b


def test_caption_cache_persists_across_captioner_instances(tmp_path):
    calls = []

    def fake_query(image_bytes: bytes) -> str:
        calls.append(image_bytes)
        return "cached across instances"

    LocalCaptioner(cache_dir=tmp_path, _query_fn=fake_query).caption(b"persistent bytes")

    # a brand-new instance, same cache_dir, same bytes -> must not re-query
    second = LocalCaptioner(cache_dir=tmp_path, _query_fn=fake_query)
    result = second.caption(b"persistent bytes")

    assert result == "cached across instances"
    assert len(calls) == 1


def test_caption_writes_a_cache_file_under_cache_dir(tmp_path):
    def fake_query(image_bytes: bytes) -> str:
        return "some caption"

    captioner = LocalCaptioner(cache_dir=tmp_path, _query_fn=fake_query)
    captioner.caption(b"bytes to cache")

    cache_files = list(tmp_path.glob("*"))
    assert len(cache_files) == 1
