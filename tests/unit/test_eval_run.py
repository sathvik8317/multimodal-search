import dataclasses

import pytest

from mmsearch.eval.dataset import Label, load_labels
from mmsearch.eval.run import EvalReport, evaluate, run_ablations, score_hit
from mmsearch.retrieve.types import SearchResult
from mmsearch.schema import Modality, TextSource


def _result(id_, modality=Modality.PDF_PAGE, text_source=TextSource.PDF_TEXT_LAYER):
    return SearchResult(
        id=id_,
        modality=modality,
        score=1.0,
        snippet="snippet",
        thumbnail_ref="thumb.png",
        source_path="source.path",
        text_source=text_source,
    )


# --- score_hit ------------------------------------------------------------------------


def test_score_hit_single_id_expected_hits():
    assert score_hit(("pdf:a.pdf#p1",), ["pdf:a.pdf#p1", "img:b.png"], k=5) is True


def test_score_hit_single_id_expected_misses():
    assert score_hit(("pdf:a.pdf#p1",), ["img:b.png", "tbl:c.csv"], k=5) is False


def test_score_hit_multi_id_expected_or_semantics_one_hit_counts():
    # Only one of several acceptable answers is returned -> still a hit (OR semantics).
    assert (
        score_hit(
            ("tbl:data/latency.csv", "pdf:specs/bench.pdf#p3"),
            ["pdf:specs/bench.pdf#p3", "img:other.png"],
            k=5,
        )
        is True
    )


def test_score_hit_multi_id_expected_none_hit():
    assert (
        score_hit(
            ("tbl:data/latency.csv", "pdf:specs/bench.pdf#p3"),
            ["img:other.png", "code:x.py#f"],
            k=5,
        )
        is False
    )


def test_score_hit_truncates_returned_ids_to_k():
    # The expected id is present but only at rank k+1 (index k) -> must NOT count.
    returned = ["img:1.png", "img:2.png", "img:3.png", "img:4.png", "img:5.png", "pdf:a.pdf#p1"]
    assert score_hit(("pdf:a.pdf#p1",), returned, k=5) is False


def test_score_hit_within_k_boundary_counts():
    returned = ["img:1.png", "img:2.png", "img:3.png", "img:4.png", "pdf:a.pdf#p1"]
    assert score_hit(("pdf:a.pdf#p1",), returned, k=5) is True


# --- evaluate(): aggregate hit rate ----------------------------------------------------


def test_evaluate_aggregate_hit_rate_hand_computed():
    labels = [
        Label(query="q1", expected=("pdf:a.pdf#p1",)),
        Label(query="q2", expected=("img:b.png",)),
        Label(query="q3", expected=("tbl:c.csv",)),
        Label(query="q4", expected=("code:d.py#f",)),
    ]
    id_index = {
        "pdf:a.pdf#p1": (Modality.PDF_PAGE, TextSource.PDF_TEXT_LAYER),
        "img:b.png": (Modality.DIAGRAM, TextSource.VLM_CAPTION),
        "tbl:c.csv": (Modality.TABLE, TextSource.TABLE_MARKDOWN),
        "code:d.py#f": (Modality.CODE, TextSource.CODE_SOURCE),
    }

    # q1 hits, q2 hits, q3 misses, q4 misses -> aggregate = 2/4 = 0.5
    canned = {
        "q1": [_result("pdf:a.pdf#p1", Modality.PDF_PAGE, TextSource.PDF_TEXT_LAYER)],
        "q2": [_result("img:b.png", Modality.DIAGRAM, TextSource.VLM_CAPTION)],
        "q3": [_result("img:other.png", Modality.DIAGRAM, TextSource.VLM_CAPTION)],
        "q4": [_result("img:other.png", Modality.DIAGRAM, TextSource.VLM_CAPTION)],
    }

    def fake_search(query: str, k: int = 5):
        return canned[query]

    report = evaluate(fake_search, labels, id_index, k=5)
    assert report.aggregate_hit_rate == 0.5


# --- evaluate(): per-modality attribution -----------------------------------------------


def test_evaluate_per_modality_worked_example_from_spec():
    # expected=("tbl:...", "pdf:...") where only the pdf id is ever returned.
    labels = [
        Label(query="q1", expected=("tbl:data/latency.csv", "pdf:specs/bench.pdf#p3")),
    ]
    id_index = {
        "tbl:data/latency.csv": (Modality.TABLE, TextSource.TABLE_MARKDOWN),
        "pdf:specs/bench.pdf#p3": (Modality.PDF_PAGE, TextSource.PDF_TEXT_LAYER),
    }

    def fake_search(query: str, k: int = 5):
        return [_result("pdf:specs/bench.pdf#p3", Modality.PDF_PAGE, TextSource.PDF_TEXT_LAYER)]

    report = evaluate(fake_search, labels, id_index, k=5)
    assert report.per_modality["table"] == 0.0
    assert report.per_modality["pdf_page"] == 1.0


def test_evaluate_per_modality_denominator_across_two_queries_numerator_one():
    # "diagram" modality listed in two queries; only one of them hits -> 1/2 = 0.5
    labels = [
        Label(query="q1", expected=("img:a.png",)),
        Label(query="q2", expected=("img:b.png",)),
    ]
    id_index = {
        "img:a.png": (Modality.DIAGRAM, TextSource.VLM_CAPTION),
        "img:b.png": (Modality.DIAGRAM, TextSource.VLM_CAPTION),
    }
    canned = {
        "q1": [_result("img:a.png", Modality.DIAGRAM, TextSource.VLM_CAPTION)],  # hit
        "q2": [_result("img:other.png", Modality.DIAGRAM, TextSource.VLM_CAPTION)],  # miss
    }

    def fake_search(query: str, k: int = 5):
        return canned[query]

    report = evaluate(fake_search, labels, id_index, k=5)
    assert report.per_modality["diagram"] == 0.5


# --- evaluate(): per-text_source attribution ---------------------------------------------


def test_evaluate_per_text_source_worked_example():
    labels = [
        Label(query="q1", expected=("tbl:data/latency.csv", "pdf:specs/bench.pdf#p3")),
    ]
    id_index = {
        "tbl:data/latency.csv": (Modality.TABLE, TextSource.TABLE_MARKDOWN),
        "pdf:specs/bench.pdf#p3": (Modality.PDF_PAGE, TextSource.PDF_TEXT_LAYER),
    }

    def fake_search(query: str, k: int = 5):
        return [_result("pdf:specs/bench.pdf#p3", Modality.PDF_PAGE, TextSource.PDF_TEXT_LAYER)]

    report = evaluate(fake_search, labels, id_index, k=5)
    assert report.per_text_source["table_markdown"] == 0.0
    assert report.per_text_source["pdf_text_layer"] == 1.0


def test_evaluate_per_text_source_denominator_across_two_queries_numerator_one():
    labels = [
        Label(query="q1", expected=("code:a.py#f",)),
        Label(query="q2", expected=("code:b.py#g",)),
    ]
    id_index = {
        "code:a.py#f": (Modality.CODE, TextSource.CODE_SOURCE),
        "code:b.py#g": (Modality.CODE, TextSource.CODE_SOURCE),
    }
    canned = {
        "q1": [_result("code:a.py#f", Modality.CODE, TextSource.CODE_SOURCE)],  # hit
        "q2": [_result("code:other.py#h", Modality.CODE, TextSource.CODE_SOURCE)],  # miss
    }

    def fake_search(query: str, k: int = 5):
        return canned[query]

    report = evaluate(fake_search, labels, id_index, k=5)
    assert report.per_text_source["code_source"] == 0.5


def test_evaluate_returns_eval_report_instance():
    labels = [Label(query="q1", expected=("pdf:a.pdf#p1",))]
    id_index = {"pdf:a.pdf#p1": (Modality.PDF_PAGE, TextSource.PDF_TEXT_LAYER)}

    def fake_search(query: str, k: int = 5):
        return []

    report = evaluate(fake_search, labels, id_index, k=5)
    assert isinstance(report, EvalReport)


def test_eval_report_is_immutable():
    report = EvalReport(aggregate_hit_rate=1.0, per_modality={}, per_text_source={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.aggregate_hit_rate = 0.0  # type: ignore[misc]


def test_evaluate_default_k_uses_config_top_k():
    from mmsearch import config

    labels = [Label(query="q1", expected=("pdf:a.pdf#p1",))]
    id_index = {"pdf:a.pdf#p1": (Modality.PDF_PAGE, TextSource.PDF_TEXT_LAYER)}

    seen_k = {}

    def fake_search(query: str, k: int = 5):
        seen_k["k"] = k
        return []

    evaluate(fake_search, labels, id_index)
    assert seen_k["k"] == config.TOP_K


# --- run_ablations() --------------------------------------------------------------------


def test_run_ablations_returns_exact_keys_and_independent_reports():
    labels = [
        Label(query="q1", expected=("pdf:a.pdf#p1",)),
        Label(query="q2", expected=("img:b.png",)),
    ]
    id_index = {
        "pdf:a.pdf#p1": (Modality.PDF_PAGE, TextSource.PDF_TEXT_LAYER),
        "img:b.png": (Modality.DIAGRAM, TextSource.VLM_CAPTION),
    }

    def always_miss(query: str, k: int = 5):
        return [_result("img:other.png", Modality.DIAGRAM, TextSource.VLM_CAPTION)]

    def always_hit(query: str, k: int = 5):
        if query == "q1":
            return [_result("pdf:a.pdf#p1", Modality.PDF_PAGE, TextSource.PDF_TEXT_LAYER)]
        return [_result("img:b.png", Modality.DIAGRAM, TextSource.VLM_CAPTION)]

    def half_hit(query: str, k: int = 5):
        if query == "q1":
            return [_result("pdf:a.pdf#p1", Modality.PDF_PAGE, TextSource.PDF_TEXT_LAYER)]
        return [_result("img:other.png", Modality.DIAGRAM, TextSource.VLM_CAPTION)]

    search_fns = {
        "vector-only": always_miss,
        "rrf-only": always_hit,
        "rrf+rerank": half_hit,
    }

    reports = run_ablations(search_fns, labels, id_index, k=5)

    assert set(reports.keys()) == {"vector-only", "rrf-only", "rrf+rerank"}
    assert reports["vector-only"].aggregate_hit_rate == 0.0
    assert reports["rrf-only"].aggregate_hit_rate == 1.0
    assert reports["rrf+rerank"].aggregate_hit_rate == 0.5


# --- integration: composes with the real dataset.load_labels ----------------------------


def test_evaluate_composes_with_real_load_labels(tmp_path):
    path = tmp_path / "labels.yaml"
    path.write_text(
        """
        - query: "the diagram showing auth token flow"
          expected: ["img:docs/auth-flow.png"]
        - query: "p99 latency numbers for the reranker"
          expected: ["tbl:data/latency.csv", "pdf:specs/bench.pdf#p3"]
        - query: "retry backoff implementation"
          expected: ["code:src/retry.py#backoff"]
        """
    )
    labels = load_labels(path)

    id_index = {
        "img:docs/auth-flow.png": (Modality.DIAGRAM, TextSource.VLM_CAPTION),
        "tbl:data/latency.csv": (Modality.TABLE, TextSource.TABLE_MARKDOWN),
        "pdf:specs/bench.pdf#p3": (Modality.PDF_PAGE, TextSource.PDF_TEXT_LAYER),
        "code:src/retry.py#backoff": (Modality.CODE, TextSource.CODE_SOURCE),
    }

    canned = {
        "the diagram showing auth token flow": [
            _result("img:docs/auth-flow.png", Modality.DIAGRAM, TextSource.VLM_CAPTION)
        ],
        "p99 latency numbers for the reranker": [
            _result("pdf:specs/bench.pdf#p3", Modality.PDF_PAGE, TextSource.PDF_TEXT_LAYER)
        ],
        "retry backoff implementation": [
            _result("code:other.py#g", Modality.CODE, TextSource.CODE_SOURCE)
        ],
    }

    def fake_search(query: str, k: int = 5):
        return canned[query]

    report = evaluate(fake_search, labels, id_index, k=5)

    # q1 hit, q2 hit, q3 miss -> 2/3
    assert report.aggregate_hit_rate == pytest.approx(2 / 3)
    assert report.per_modality["diagram"] == 1.0
    assert report.per_modality["table"] == 0.0
    assert report.per_modality["pdf_page"] == 1.0
    assert report.per_modality["code"] == 0.0
