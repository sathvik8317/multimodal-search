import pytest

from mmsearch import config
from mmsearch.clients.fakes import FakeEmbeddingClient, FakeReranker
from mmsearch.clients.protocols import EmbedInput, RerankResult
from mmsearch.db import ensure_fts_index, open_table, upsert
from mmsearch.retrieve.pipeline import build_search_fn
from mmsearch.retrieve.types import SearchResult
from mmsearch.schema import Modality, Row, TextSource

EMBEDDER = FakeEmbeddingClient()


def _row(id_: str, content_text: str, modality: Modality, **overrides) -> Row:
    defaults = dict(
        id=id_,
        modality=modality,
        content_text=content_text,
        text_source=TextSource.CODE_SOURCE,
        vector=EMBEDDER.embed_documents([EmbedInput(text=content_text)])[0],
        source_path="src/a.py",
    )
    defaults.update(overrides)
    return Row(**defaults)


@pytest.fixture
def table(tmp_path):
    table = open_table(uri=tmp_path)
    rows = [
        _row("code:a.py#f", "the retry backoff is exponential", Modality.CODE),
        _row("pdf:doc.pdf#p1", "authentication token flow diagram overview", Modality.PDF_PAGE,
             text_source=TextSource.PDF_TEXT_LAYER),
        _row("img:auth.png", "diagram showing the auth token flow", Modality.DIAGRAM,
             text_source=TextSource.VLM_CAPTION, thumbnail_ref="auth.png"),
        _row("tbl:latency.csv", "p99 latency numbers for the reranker service", Modality.TABLE,
             text_source=TextSource.TABLE_MARKDOWN),
        _row("code:b.py#g", "completely unrelated gardening content", Modality.CODE),
    ]
    upsert(table, rows)
    ensure_fts_index(table)
    return table


class RaisingReranker:
    def rerank(self, query, documents, top_n):
        raise RuntimeError("boom")


class NeverCallReranker:
    def rerank(self, query, documents, top_n):
        raise AssertionError("reranker should not be called in this mode")


class RecordingRetrieverTable:
    """Wraps a real table, records which query_types were searched."""

    def __init__(self, inner):
        self._inner = inner
        self.query_types_used = []

    def search(self, query, query_type):
        self.query_types_used.append(query_type)
        return self._inner.search(query, query_type=query_type)


# --- rrf+rerank mode (default) -----------------------------------------------------------

def test_rrf_rerank_returns_up_to_k_results_of_right_shape(table):
    search_fn = build_search_fn(table, EMBEDDER, FakeReranker(), mode="rrf+rerank")

    results = search_fn("auth token flow diagram", k=3)

    assert 0 < len(results) <= 3
    for result in results:
        assert isinstance(result, SearchResult)
        assert isinstance(result.modality, Modality)
        assert isinstance(result.text_source, TextSource)
        assert isinstance(result.score, float)
        assert isinstance(result.snippet, str)


def test_rrf_rerank_falls_back_to_rrf_order_when_reranker_raises(table):
    search_fn = build_search_fn(table, EMBEDDER, RaisingReranker(), mode="rrf+rerank")

    # Should not raise, despite the reranker always raising.
    results = search_fn("auth token flow diagram", k=3)

    assert len(results) > 0


def test_rrf_rerank_falls_back_when_reranker_is_none(table):
    search_fn = build_search_fn(table, EMBEDDER, None, mode="rrf+rerank")

    results = search_fn("auth token flow diagram", k=3)

    assert len(results) > 0


# --- vector-only mode ----------------------------------------------------------------------

def test_vector_only_never_calls_fts_or_reranker(table):
    search_fn = build_search_fn(table, EMBEDDER, NeverCallReranker(), mode="vector-only")

    results = search_fn("auth token flow diagram", k=3)

    assert 0 < len(results) <= 3
    # NeverCallReranker would have raised AssertionError if invoked; since we
    # got here without exception, it was never called.


def test_vector_only_does_not_invoke_fts_search(table, monkeypatch):
    original_search = type(table).search
    calls = []

    def spy_search(self, query, query_type=None, *args, **kwargs):
        calls.append(query_type)
        return original_search(self, query, query_type=query_type, *args, **kwargs)

    monkeypatch.setattr(type(table), "search", spy_search)

    search_fn = build_search_fn(table, EMBEDDER, FakeReranker(), mode="vector-only")
    search_fn("auth token flow diagram", k=3)

    assert "fts" not in calls
    assert "vector" in calls


# --- rrf-only mode -------------------------------------------------------------------------

def test_rrf_only_calls_both_retrievers_but_never_reranker(table, monkeypatch):
    original_search = type(table).search
    calls = []

    def spy_search(self, query, query_type=None, *args, **kwargs):
        calls.append(query_type)
        return original_search(self, query, query_type=query_type, *args, **kwargs)

    monkeypatch.setattr(type(table), "search", spy_search)

    search_fn = build_search_fn(table, EMBEDDER, NeverCallReranker(), mode="rrf-only")
    results = search_fn("auth token flow diagram", k=3)

    assert "fts" in calls
    assert "vector" in calls
    assert 0 < len(results) <= 3
