import json

import pytest
from fastapi import HTTPException

from mmsearch import config
from mmsearch.clients.fakes import FakeEmbeddingClient, FakeReranker
from mmsearch.clients.protocols import EmbedInput, RerankResult
from mmsearch.db import ensure_fts_index, open_table, upsert
from mmsearch.retrieve.pipeline import _rerank_text, _row_to_result, build_search_fn
from mmsearch.retrieve.types import SearchResult
from mmsearch.schema import Modality, Row, TextSource

COHERE_EMBEDDER = FakeEmbeddingClient(dim=config.COHERE_EMBED_DIM)
OPENAI_EMBEDDER = FakeEmbeddingClient(dim=config.OPENAI_EMBED_DIM)


def _row(id_: str, content_text: str, modality: Modality, **overrides) -> Row:
    """Populate vectors the way real ingestion does (post Phase 3): every row
    gets an OpenAI text vector; pdf_page/diagram also get a Cohere image
    vector. Before Phase 3, text-layer pdf_page rows had no vector_openai --
    ingest/documents.py's ingest_pdf now embeds every page's text regardless
    of text_source, batched per-document.
    """
    text_source = overrides.get("text_source", TextSource.CODE_SOURCE)
    vector_cohere = None
    if modality in (Modality.PDF_PAGE, Modality.DIAGRAM):
        vector_cohere = COHERE_EMBEDDER.embed_documents([EmbedInput(text=content_text)])[0]
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

    def spy_search(self, query=None, query_type=None, *args, **kwargs):
        calls.append((query_type, kwargs.get("vector_column_name")))
        return original_search(self, query, query_type=query_type, *args, **kwargs)

    monkeypatch.setattr(type(table), "search", spy_search)

    search_fn = build_search_fn(table, COHERE_EMBEDDER, OPENAI_EMBEDDER, FakeReranker(), mode="vector-only")
    search_fn("auth token flow diagram", k=3)

    query_types = [c[0] for c in calls]
    vector_columns = {c[1] for c in calls if c[0] == "vector"}
    assert "fts" not in query_types
    assert vector_columns == {"vector_cohere", "vector_openai"}


class _ScriptedTable:
    """Fake table returning pre-scripted per-list ranked orders instead of
    real similarity search, so a specific ranking scenario can be built
    exactly -- FakeEmbeddingClient's hash-based vectors can't be steered
    precisely enough to hand-construct one. Used only for the
    eligibility-normalization integration test below.
    """

    def __init__(self, rows_by_id, cohere_ranking, openai_ranking, cohere_eligible, openai_eligible):
        self._rows_by_id = rows_by_id
        self._cohere_ranking = cohere_ranking
        self._openai_ranking = openai_ranking
        self._cohere_eligible = cohere_eligible
        self._openai_eligible = openai_eligible

    def search(self, query=None, query_type=None, vector_column_name=None):
        if query_type is None and vector_column_name is None:
            # _build_eligibility's table.search().select([...]).to_list()
            rows = [
                {
                    "id": id_,
                    "vector_cohere": [0.0] if id_ in self._cohere_eligible else None,
                    "vector_openai": [0.0] if id_ in self._openai_eligible else None,
                }
                for id_ in self._rows_by_id
            ]
            return _Chain(rows)
        if vector_column_name == "vector_cohere":
            ranking = self._cohere_ranking
        elif vector_column_name == "vector_openai":
            ranking = self._openai_ranking
        else:
            raise AssertionError("this test never issues an FTS search")
        return _Chain([self._rows_by_id[id_] for id_ in ranking])


class _Chain:
    def __init__(self, rows):
        self._rows = rows

    def select(self, columns):
        return self

    def limit(self, n):
        return _Chain(self._rows[:n])

    def to_list(self):
        return self._rows


def test_vector_only_no_longer_lets_a_multi_eligible_row_beat_a_better_ranked_single_eligible_row():
    """Integration check for the eligibility-normalization fix, exercised
    through build_search_fn end-to-end (not just fusion.py's pure-function
    tests) -- traces the real bug found against the committed corpus: a code
    row ranking #1 in its only eligible list (OpenAI) must beat a pdf_page
    row that merely appears in *both* lists at worse individual ranks.
    Plain sum-RRF got this backwards (see the k=1 hand computation below,
    mirroring fusion.py's test_multi_eligible_list_id_is_normalized...).
    """
    rows_by_id = {
        "code:winner": {
            "id": "code:winner",
            "modality": "code",
            "content_text": "def winner(): pass",
            "thumbnail_ref": "",
            "source_path": "src/winner.py",
            "text_source": "code_source",
        },
        "pdf:loser": {
            "id": "pdf:loser",
            "modality": "pdf_page",
            "content_text": "irrelevant content",
            "thumbnail_ref": "loser.png",
            "source_path": "specs/loser.pdf",
            "text_source": "pdf_text_layer",
        },
    }
    # k=1: code:winner (openai rank0) = 1.0, denominator 1 (openai-only eligible) -> 1.0.
    # pdf:loser (cohere rank0=1.0, openai rank1=0.5) sums to 1.5 under plain
    # sum-RRF -- beating code:winner despite being ranked worse on average.
    # Eligibility-normalized: 1.5 / 2 eligible lists = 0.75, correctly below
    # code:winner's 1.0.
    scripted_table = _ScriptedTable(
        rows_by_id=rows_by_id,
        cohere_ranking=["pdf:loser"],
        openai_ranking=["code:winner", "pdf:loser"],
        cohere_eligible={"pdf:loser"},
        openai_eligible={"code:winner", "pdf:loser"},
    )

    search_fn = build_search_fn(
        scripted_table, COHERE_EMBEDDER, OPENAI_EMBEDDER, NeverCallReranker(), mode="vector-only", rrf_k=1
    )
    results = search_fn("anything", k=2)

    assert [r.id for r in results] == ["code:winner", "pdf:loser"]


# --- rrf-only mode -------------------------------------------------------------------------

def test_rrf_only_calls_both_vector_retrievers_and_fts_but_never_reranker(table, monkeypatch):
    original_search = type(table).search
    calls = []

    def spy_search(self, query=None, query_type=None, *args, **kwargs):
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

    def spy_search(self, query=None, query_type=None, *args, **kwargs):
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


# --- _rerank_text: table reranking miscalibration fix -----------------------------------
#
# Diagnostic finding: the full ~12KB markdown table blob sent as rerank input scored
# 0.39-0.66 on completely unrelated queries (Kubernetes/lasagna/quarterly-earnings style),
# while PDF prose controls correctly scored near-zero on the same queries (3-20x ratio).
# _rerank_text() sends a short synthetic summary for table rows instead of the raw blob;
# embedding and FTS still use content_text unchanged (see build_search_fn -- id_to_row's
# content_text is untouched, only the rerank shortlist_docs construction changed).

def _table_row_dict(columns, data_rows, total_rows, source_path="data/example.csv"):
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator] + ["| " + " | ".join(row) + " |" for row in data_rows]
    content_text = "\n".join(lines)
    metadata = json.dumps(
        {
            "n_rows": len(data_rows),
            "n_cols": len(columns),
            "columns": columns,
            "truncated": False,
            "total_rows": total_rows,
        }
    )
    return {
        "id": f"tbl:{source_path}",
        "modality": "table",
        "content_text": content_text,
        "thumbnail_ref": "",
        "source_path": source_path,
        "text_source": "table_markdown",
        "metadata": metadata,
    }


def test_rerank_text_passthrough_for_non_table_modalities():
    row = _code_row(_CODE_TEXT_FUNCTION)
    assert _rerank_text(row) == row["content_text"]


def test_rerank_text_for_table_row_is_much_shorter_than_content_text():
    row = _table_row_dict(
        columns=["Make", "Model", "Year"],
        data_rows=[["Toyota", "Camry", "2020"], ["Honda", "Civic", "2019"]] * 20,
        total_rows=5000,
    )
    summary = _rerank_text(row)
    assert len(summary) < len(row["content_text"])
    assert len(summary) <= 500


def test_rerank_text_for_table_row_includes_filename_columns_and_row_count():
    row = _table_row_dict(
        columns=["Make", "Model", "Year"],
        data_rows=[["Toyota", "Camry", "2020"]],
        total_rows=5000,
        source_path="data/car_prediction_data.csv",
    )
    summary = _rerank_text(row)
    assert "car_prediction_data.csv" in summary
    assert "Make" in summary and "Model" in summary and "Year" in summary
    assert "5000" in summary


def test_rerank_text_for_table_row_includes_a_data_sample():
    row = _table_row_dict(
        columns=["Make", "Model"],
        data_rows=[["Toyota", "Camry"], ["Honda", "Civic"]],
        total_rows=2,
    )
    summary = _rerank_text(row)
    assert "Toyota" in summary or "Camry" in summary


def test_rerank_text_for_table_row_is_capped_even_with_long_column_names():
    many_columns = [f"very_long_column_name_number_{i}" for i in range(50)]
    row = _table_row_dict(columns=many_columns, data_rows=[["x"] * 50], total_rows=1)
    summary = _rerank_text(row)
    assert len(summary) <= 500


def test_rrf_rerank_uses_rerank_text_not_raw_content_text_for_table_rows(table, monkeypatch):
    # The reranker must receive the summary, not the full markdown blob --
    # spy on what's actually passed to reranker.rerank().
    seen_docs = []

    class SpyReranker:
        def rerank(self, query, documents, top_n):
            seen_docs.extend(documents)
            return FakeReranker().rerank(query, documents, top_n)

    search_fn = build_search_fn(table, COHERE_EMBEDDER, OPENAI_EMBEDDER, SpyReranker(), mode="rrf+rerank")
    search_fn("p99 latency numbers for the reranker service", k=3)

    table_row = table.to_arrow().to_pylist()
    table_content_text = next(r["content_text"] for r in table_row if r["modality"] == "table")
    assert table_content_text not in seen_docs  # raw blob never reaches the reranker


# --- live: proves the fix against the real Cohere API (opt-in, `pytest -m live`) --------
#
# Not a fake -- the whole point is verifying real Cohere Rerank behavior against the
# actual committed corpus, the same direct-rerank-call approach the diagnostic used.
# Excluded from the default suite by pyproject.toml's `addopts = -m "not live"`.

@pytest.mark.live
def test_table_rerank_summary_scores_near_zero_on_unrelated_queries_like_pdf_controls():
    from mmsearch import db
    from mmsearch.clients.cohere import CohereClient

    real_table = db.open_table()
    rows = real_table.to_arrow().to_pylist()
    table_rows = [r for r in rows if r["modality"] == "table"]
    assert len(table_rows) == 4  # sanity: the real committed corpus

    pdf_ids = [
        "pdf:specs/2407.01449v6.pdf#p1",
        "pdf:specs/Fine-Tune_Your_Own_LLM_for_Free_on_a_Kaggle_GPU_in_30_Minutes.pdf#p1",
        "pdf:specs/2407.01449v6.pdf#p12",
    ]
    pdf_rows = [next(r for r in rows if r["id"] == pid) for pid in pdf_ids]

    docs = table_rows + pdf_rows
    doc_texts = [_rerank_text(r) for r in docs]

    # the fix must not touch content_text used for embedding/FTS
    for row, text in zip(table_rows, doc_texts[: len(table_rows)]):
        assert text != row["content_text"]
        assert len(text) <= 500

    client = CohereClient()
    # "quarterly earnings" is deliberately not treated the same as the other
    # two: ecommerce_sales_analytics_5000.csv genuinely has a "revenue"
    # column, so post-fix it picks up a real (if weak) lexical connection to
    # "earnings" -- measured at 0.139, vs. 0.53-0.71 pre-fix for the exact
    # same query/table pair. That is the reranker doing its job now that it
    # has real content to judge, not the bug (which was topic-blind: every
    # query scored high regardless of any actual relevance). "Kubernetes
    # ingress" and "lasagna recipe" have no such legitimate connection to
    # any table, so those two are held to the tight near-zero bound that
    # directly proves the bug is gone.
    tight_queries = ["Kubernetes ingress", "lasagna recipe"]
    loose_queries = ["quarterly earnings"]

    for query in tight_queries + loose_queries:
        results = client.rerank(query, doc_texts, top_n=len(doc_texts))
        by_index = {r.index: r.relevance_score for r in results}
        table_scores = [by_index[i] for i in range(len(table_rows))]
        bound = 0.1 if query in tight_queries else 0.2
        for score in table_scores:
            assert score < bound, (
                f"table rerank score {score} for query {query!r} exceeds {bound} after "
                "the summary fix -- still elevated, not just a weak genuine lexical match"
            )
