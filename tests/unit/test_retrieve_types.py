import dataclasses

import pytest

from mmsearch.retrieve.types import SearchFn, SearchResult
from mmsearch.schema import Modality, TextSource


def _make_result(**overrides):
    defaults = dict(
        id="pdf:specs/rfc.pdf#p14",
        modality=Modality.PDF_PAGE,
        score=0.87,
        snippet="The retry backoff is exponential with base 2...",
        thumbnail_ref="specs/rfc.pdf/p14.png",
        source_path="specs/rfc.pdf",
        text_source=TextSource.PDF_TEXT_LAYER,
    )
    defaults.update(overrides)
    return SearchResult(**defaults)


def test_search_result_holds_all_fields():
    result = _make_result()
    assert result.id == "pdf:specs/rfc.pdf#p14"
    assert result.modality == Modality.PDF_PAGE
    assert result.score == 0.87
    assert result.snippet.startswith("The retry backoff")
    assert result.thumbnail_ref == "specs/rfc.pdf/p14.png"
    assert result.source_path == "specs/rfc.pdf"
    assert result.text_source == TextSource.PDF_TEXT_LAYER


def test_search_result_is_immutable():
    result = _make_result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.score = 0.99  # type: ignore[misc]


def test_search_fn_protocol_matches_a_conforming_callable():
    def fake_search(query: str, k: int = 5) -> list[SearchResult]:
        return [_make_result()]

    fn: SearchFn = fake_search
    results = fn("what's the retry backoff", k=5)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)


def test_search_fn_default_k_is_usable():
    def fake_search(query: str, k: int = 5) -> list[SearchResult]:
        assert k == 5
        return []

    fn: SearchFn = fake_search
    assert fn("query without explicit k") == []
