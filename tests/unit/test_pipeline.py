import pytest
from fastapi import HTTPException

from mmsearch import config
from mmsearch.clients.fakes import FakeEmbeddingClient, FakeReranker
from mmsearch.clients.protocols import EmbedInput, RerankResult
from mmsearch.db import ensure_fts_index, open_table, upsert
from mmsearch.retrieve.pipeline import _row_to_result, build_search_fn
from mmsearch.retrieve.types import SearchResult
from mmsearch.schema import Modality, Row, TextSource

COHERE_EMBEDDER = FakeEmbeddingClient(dim=config.COHERE_EMBED_DIM)
OPENAI_EMBEDDER = FakeEmbeddingClient(dim=config.OPENAI_EMBED_DIM)


def _row(id_: str, content_text: str, modality: Modality, **overrides) -> Row:
    """Populate vectors the way real ingestion does: pdf_page/diagram get a
    Cohere image vector (diagram/scanned-caption pages also get an OpenAI
    caption vector); table/code get an OpenAI text vector only.
    """
    text_source = overrides.get("text_source", TextSource.CODE_SOURCE)
    vector_cohere = None
    vector_openai = None
    if modality in (Modality.PDF_PAGE, Modality.DIAGRAM):
        vector_cohere = COHERE_EMBEDDER.embed_documents([EmbedInput(text=content_text)])[0]
    if modality in (Modality.TABLE, Modality.CODE) or text_source == TextSource.VLM_CAPTION:
        vector_openai = OPENAI_EMBEDDER.embed_documents([EmbedInput(text=content_text)])[0]

    defaults = dict(
        id=id_,
        modality=modality,
        content_text=content_text,
        text_source=text_source,
        vector_cohere=vector_cohere,
        vector_openai=vector_openai,
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


class RaisingEmbeddingClient:
    def embed_query(self, text):
        raise RuntimeError("embedder is down")

    def embed_documents(self, items):
        raise RuntimeError("embedder is down")


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
    search_fn = build_search_fn(table, COHERE_EMBEDDER, OPENAI_EMBEDDER, FakeReranker(), mode="rrf+rerank")

    results = search_fn("auth token flow diagram", k=3)

    assert 0 < len(results) <= 3
    for result in results:
        assert isinstance(result, SearchResult)
        assert isinstance(result.modality, Modality)
        assert isinstance(result.text_source, TextSource)
        assert isinstance(result.score, float)
        assert isinstance(result.snippet, str)


def test_rrf_rerank_falls_back_to_rrf_order_when_reranker_raises(table):
    search_fn = build_search_fn(table, COHERE_EMBEDDER, OPENAI_EMBEDDER, RaisingReranker(), mode="rrf+rerank")

    # Should not raise, despite the reranker always raising.
    results = search_fn("auth token flow diagram", k=3)

    assert len(results) > 0


def test_rrf_rerank_falls_back_when_reranker_is_none(table):
    search_fn = build_search_fn(table, COHERE_EMBEDDER, OPENAI_EMBEDDER, None, mode="rrf+rerank")

    results = search_fn("auth token flow diagram", k=3)

    assert len(results) > 0


# --- vector-only mode ----------------------------------------------------------------------

def test_vector_only_never_calls_fts_or_reranker(table):
    search_fn = build_search_fn(table, COHERE_EMBEDDER, OPENAI_EMBEDDER, NeverCallReranker(), mode="vector-only")

    results = search_fn("auth token flow diagram", k=3)

    assert 0 < len(results) <= 3
    # NeverCallReranker would have raised AssertionError if invoked; since we
    # got here without exception, it was never called.


def test_vector_only_does_not_invoke_fts_search(table, monkeypatch):
    original_search = type(table).search
    calls = []

    def spy_search(self, query, query_type=None, *args, **kwargs):
        calls.append((query_type, kwargs.get("vector_column_name")))
        return original_search(self, query, query_type=query_type, *args, **kwargs)

    monkeypatch.setattr(type(table), "search", spy_search)

    search_fn = build_search_fn(table, COHERE_EMBEDDER, OPENAI_EMBEDDER, FakeReranker(), mode="vector-only")
    search_fn("auth token flow diagram", k=3)

    query_types = [c[0] for c in calls]
    vector_columns = {c[1] for c in calls if c[0] == "vector"}
    assert "fts" not in query_types
    assert vector_columns == {"vector_cohere", "vector_openai"}


# --- rrf-only mode -------------------------------------------------------------------------

def test_rrf_only_calls_both_vector_retrievers_and_fts_but_never_reranker(table, monkeypatch):
    original_search = type(table).search
    calls = []

    def spy_search(self, query, query_type=None, *args, **kwargs):
        calls.append((query_type, kwargs.get("vector_column_name")))
        return original_search(self, query, query_type=query_type, *args, **kwargs)

    monkeypatch.setattr(type(table), "search", spy_search)

    search_fn = build_search_fn(table, COHERE_EMBEDDER, OPENAI_EMBEDDER, NeverCallReranker(), mode="rrf-only")
    results = search_fn("auth token flow diagram", k=3)

    query_types = [c[0] for c in calls]
    vector_columns = {c[1] for c in calls if c[0] == "vector"}
    assert "fts" in query_types
    assert vector_columns == {"vector_cohere", "vector_openai"}
    assert 0 < len(results) <= 3


# --- embedder failure handling (Bug 1) --------------------------------------------------------
#
# Mirrors the reranker fallback pattern above: a failing retriever must never
# surface a 500 -- it's dropped from RRF instead. Only vector-only mode has
# nothing left to fall back to when BOTH embedders fail (no FTS call in that
# mode), so that specific case raises a clean 503 instead of silently
# returning an empty list.

def test_cohere_embedder_failure_falls_back_to_openai_only(table):
    search_fn = build_search_fn(
        table, RaisingEmbeddingClient(), OPENAI_EMBEDDER, FakeReranker(), mode="rrf+rerank"
    )

    results = search_fn("auth token flow diagram", k=3)  # must not raise

    assert len(results) > 0


def test_openai_embedder_failure_falls_back_to_cohere_only(table):
    search_fn = build_search_fn(
        table, COHERE_EMBEDDER, RaisingEmbeddingClient(), FakeReranker(), mode="rrf+rerank"
    )

    results = search_fn("auth token flow diagram", k=3)  # must not raise

    assert len(results) > 0


def test_cohere_embedder_failure_skips_the_cohere_vector_search_call(table, monkeypatch):
    original_search = type(table).search
    calls = []

    def spy_search(self, query, query_type=None, *args, **kwargs):
        calls.append((query_type, kwargs.get("vector_column_name")))
        return original_search(self, query, query_type=query_type, *args, **kwargs)

    monkeypatch.setattr(type(table), "search", spy_search)

    search_fn = build_search_fn(
        table, RaisingEmbeddingClient(), OPENAI_EMBEDDER, FakeReranker(), mode="rrf+rerank"
    )
    search_fn("auth token flow diagram", k=3)

    vector_columns = {c[1] for c in calls if c[0] == "vector"}
    assert vector_columns == {"vector_openai"}  # vector_cohere never attempted


def test_both_embedders_fail_in_vector_only_mode_raises_503(table):
    search_fn = build_search_fn(
        table, RaisingEmbeddingClient(), RaisingEmbeddingClient(), FakeReranker(), mode="vector-only"
    )

    with pytest.raises(HTTPException) as exc_info:
        search_fn("auth token flow diagram", k=3)
    assert exc_info.value.status_code == 503


def test_both_embedders_fail_in_rrf_only_mode_falls_back_to_fts_only(table):
    search_fn = build_search_fn(
        table, RaisingEmbeddingClient(), RaisingEmbeddingClient(), NeverCallReranker(), mode="rrf-only"
    )

    # Must not raise -- FTS still runs even though both vector retrievers failed.
    results = search_fn("auth token flow diagram", k=3)

    assert len(results) > 0


def test_both_embedders_fail_in_rrf_rerank_mode_falls_back_to_fts_only(table):
    search_fn = build_search_fn(
        table, RaisingEmbeddingClient(), RaisingEmbeddingClient(), FakeReranker(), mode="rrf+rerank"
    )

    results = search_fn("auth token flow diagram", k=3)  # must not raise

    assert len(results) > 0


def test_one_embedder_failing_does_not_raise_in_vector_only_mode(table):
    search_fn = build_search_fn(
        table, RaisingEmbeddingClient(), OPENAI_EMBEDDER, FakeReranker(), mode="vector-only"
    )

    results = search_fn("auth token flow diagram", k=3)  # must not raise, must not 503

    assert len(results) > 0


# --- min_score_threshold -----------------------------------------------------------------------

def test_default_threshold_matches_config_and_is_a_noop(table):
    # Omitting min_score_threshold must behave identically to passing
    # config.MIN_SCORE_THRESHOLD explicitly -- that's the whole point of a
    # "safe default" for a parameter that wires straight into config.py.
    default_fn = build_search_fn(table, COHERE_EMBEDDER, OPENAI_EMBEDDER, FakeReranker(), mode="rrf+rerank")
    explicit_fn = build_search_fn(
        table, COHERE_EMBEDDER, OPENAI_EMBEDDER, FakeReranker(), mode="rrf+rerank",
        min_score_threshold=config.MIN_SCORE_THRESHOLD,
    )

    default_results = default_fn("auth token flow diagram", k=3)
    explicit_results = explicit_fn("auth token flow diagram", k=3)

    assert [r.id for r in default_results] == [r.id for r in explicit_results]
    assert len(default_results) > 0


def test_threshold_filters_out_low_scoring_positional_results(table):
    # rrf-only mode never reranks, so scores are purely positional
    # (1/(rank+1)): rank0=1.0, rank1=0.5, rank2=0.333... A 0.4 threshold
    # keeps only the top two positions.
    search_fn = build_search_fn(
        table, COHERE_EMBEDDER, OPENAI_EMBEDDER, FakeReranker(), mode="rrf-only",
        min_score_threshold=0.4,
    )

    results = search_fn("auth token flow diagram", k=5)

    assert len(results) <= 2
    for result in results:
        assert result.score >= 0.4


def test_threshold_returns_empty_list_when_no_result_exceeds_it(table):
    # No real score (positional or Cohere relevance_score) ever reaches 2.0.
    search_fn = build_search_fn(
        table, COHERE_EMBEDDER, OPENAI_EMBEDDER, FakeReranker(), mode="rrf+rerank",
        min_score_threshold=2.0,
    )

    results = search_fn("auth token flow diagram", k=5)

    assert results == []


def test_threshold_applies_to_reranked_relevance_scores(table):
    # FakeReranker scores by query/document token overlap ratio (see
    # clients/fakes.py) -- a query sharing zero tokens with a document
    # scores exactly 0.0. Any positive threshold must drop it.
    search_fn = build_search_fn(
        table, COHERE_EMBEDDER, OPENAI_EMBEDDER, FakeReranker(), mode="rrf+rerank",
        min_score_threshold=0.01,
    )

    results = search_fn("auth token flow diagram", k=5)

    for result in results:
        assert result.score >= 0.01


def test_threshold_does_not_apply_when_vector_only_mode_raises_503(table):
    # The 503 path (both embedders down, see Bug 1 fix) fires before any
    # result list -- or threshold filtering -- exists, so this must still
    # raise regardless of min_score_threshold.
    from fastapi import HTTPException

    search_fn = build_search_fn(
        table, RaisingEmbeddingClient(), RaisingEmbeddingClient(), FakeReranker(), mode="vector-only",
        min_score_threshold=0.0,
    )

    with pytest.raises(HTTPException) as exc_info:
        search_fn("auth token flow diagram", k=3)
    assert exc_info.value.status_code == 503


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
