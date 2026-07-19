import pytest

from mmsearch import config
from mmsearch.clients.fakes import FakeEmbeddingClient, FakeReranker
from mmsearch.clients.protocols import EmbedInput, RerankResult
from mmsearch.db import ensure_fts_index, open_table, upsert
from mmsearch.retrieve.pipeline import _row_to_result, build_search_fn
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


# --- snippet construction (modality-aware) ---------------------------------------------------

def _table_row(content_text: str) -> dict:
    return {
        "id": "tbl:data/car.csv",
        "modality": "table",
        "content_text": content_text,
        "thumbnail_ref": "",
        "source_path": "data/car.csv",
        "text_source": "table_markdown",
    }


def _code_row(content_text: str, id_: str = "code:src/extractor.py#dedupe_preserve_order") -> dict:
    return {
        "id": id_,
        "modality": "code",
        "content_text": content_text,
        "thumbnail_ref": "",
        "source_path": "src/extractor.py",
        "text_source": "code_source",
    }


_TABLE_MARKDOWN = (
    "| Car_Name | Year | Selling_Price | Present_Price | Kms_Driven | Fuel_Type | Seller_Type | Transmission | Owner |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    "| ritz | 2014 | 3.35 | 5.59 | 27000 | Petrol | Dealer | Manual | 0 |\n"
    "| sx4 | 2013 | 4.75 | 9.54 | 43000 | Diesel | Dealer | Manual | 0 |\n"
    "| ciaz | 2017 | 7.25 | 9.85 | 6900 | Petrol | Dealer | Manual | 0 |\n"
    "| wagon r | 2011 | 2.85 | 4.15 | 5200 | Petrol | Dealer | Manual | 0 |\n"
)

_CODE_TEXT_FUNCTION = (
    "# file: src/extractor.py\n"
    "# language: python\n"
    "def dedupe_preserve_order(values):\n"
    "    seen = set()\n"
    "    return [v for v in values if not (v in seen or seen.add(v))]\n"
)

_CODE_TEXT_METHOD = (
    "# file: src/ingest/base.py\n"
    "# language: python\n"
    "# class: PdfIngester\n"
    "def rasterize(self, page):\n"
    "    return page.get_pixmap()\n"
)


def test_table_snippet_contains_at_least_three_full_data_rows():
    result = _row_to_result(_table_row(_TABLE_MARKDOWN), score=1.0)

    lines = result.snippet.splitlines()
    data_rows = lines[2:]  # skip header + separator
    assert len(data_rows) >= 3
    for line in data_rows:
        # no mid-row cut: every data row line ends with the closing pipe
        assert line.rstrip().endswith("|")


def test_table_snippet_header_and_separator_intact():
    result = _row_to_result(_table_row(_TABLE_MARKDOWN), score=1.0)

    lines = result.snippet.splitlines()
    assert lines[0].startswith("| Car_Name")
    assert set(lines[1].replace("|", "").strip()) <= {"-", " "}


def test_code_snippet_starts_with_def_not_comment_header():
    result = _row_to_result(_code_row(_CODE_TEXT_FUNCTION), score=1.0)

    assert result.snippet.lstrip().startswith("def ")
    assert "# file:" not in result.snippet
    assert "# language:" not in result.snippet


def test_code_snippet_for_a_method_skips_the_three_line_header():
    # Methods get a 3-line header (file, language, class) from ingest/code.py,
    # not 2 -- the snippet must still land on the def line, not "# class: ...".
    result = _row_to_result(
        _code_row(_CODE_TEXT_METHOD, id_="code:src/ingest/base.py#PdfIngester.rasterize"),
        score=1.0,
    )

    assert result.snippet.lstrip().startswith("def ")
    assert "# class:" not in result.snippet


def test_pdf_snippet_behavior_is_unchanged_flat_200_char_slice():
    long_text = "A" * 300
    row = {
        "id": "pdf:doc.pdf#p1",
        "modality": "pdf_page",
        "content_text": long_text,
        "thumbnail_ref": "",
        "source_path": "doc.pdf",
        "text_source": "pdf_text_layer",
    }

    result = _row_to_result(row, score=1.0)

    assert result.snippet == long_text[:200]
    assert len(result.snippet) == 200


def test_diagram_snippet_behavior_is_unchanged_flat_200_char_slice():
    long_text = "B" * 300
    row = {
        "id": "img:diagram.png",
        "modality": "diagram",
        "content_text": long_text,
        "thumbnail_ref": "diagram.png",
        "source_path": "diagram.png",
        "text_source": "vlm_caption",
    }

    result = _row_to_result(row, score=1.0)

    assert result.snippet == long_text[:200]
