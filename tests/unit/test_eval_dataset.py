import dataclasses

import pytest

from mmsearch.eval.dataset import Label, load_labels, validate_labels


def _write(tmp_path, text: str):
    path = tmp_path / "labels.yaml"
    path.write_text(text)
    return path


# --- load_labels --------------------------------------------------------------------

def test_load_labels_parses_single_id_expected(tmp_path):
    path = _write(
        tmp_path,
        """
        - query: "the diagram showing auth token flow"
          expected: ["img:docs/auth-flow.png"]
        """,
    )
    labels = load_labels(path)
    assert labels == [Label(query="the diagram showing auth token flow", expected=("img:docs/auth-flow.png",))]


def test_load_labels_parses_multi_id_expected_list(tmp_path):
    path = _write(
        tmp_path,
        """
        - query: "p99 latency numbers for the reranker"
          expected: ["tbl:data/latency.csv", "pdf:specs/bench.pdf#p3"]
        """,
    )
    labels = load_labels(path)
    assert labels[0].expected == ("tbl:data/latency.csv", "pdf:specs/bench.pdf#p3")


def test_load_labels_parses_multiple_entries(tmp_path):
    path = _write(
        tmp_path,
        """
        - query: "query one"
          expected: ["code:a.py#f"]
        - query: "query two"
          expected: ["tbl:data.csv"]
        """,
    )
    labels = load_labels(path)
    assert len(labels) == 2
    assert [label.query for label in labels] == ["query one", "query two"]


def test_load_labels_empty_file_returns_empty_list(tmp_path):
    path = _write(tmp_path, "")
    assert load_labels(path) == []


def test_load_labels_rejects_missing_query_key(tmp_path):
    path = _write(tmp_path, '- expected: ["code:a.py#f"]')
    with pytest.raises(ValueError, match="query"):
        load_labels(path)


def test_load_labels_rejects_missing_expected_key(tmp_path):
    path = _write(tmp_path, '- query: "hello"')
    with pytest.raises(ValueError, match="expected"):
        load_labels(path)


def test_load_labels_rejects_empty_query_string(tmp_path):
    path = _write(tmp_path, '- query: ""\n  expected: ["code:a.py#f"]')
    with pytest.raises(ValueError, match="query"):
        load_labels(path)


def test_load_labels_rejects_empty_expected_list(tmp_path):
    path = _write(tmp_path, '- query: "hello"\n  expected: []')
    with pytest.raises(ValueError, match="expected"):
        load_labels(path)


# --- Label dataclass --------------------------------------------------------------------

def test_label_is_immutable():
    label = Label(query="q", expected=("code:a.py#f",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        label.query = "changed"  # type: ignore[misc]


# --- validate_labels (typo guard) --------------------------------------------------------

def test_validate_labels_passes_when_all_ids_known():
    labels = [Label(query="q1", expected=("code:a.py#f", "tbl:data.csv"))]
    validate_labels(labels, valid_ids={"code:a.py#f", "tbl:data.csv", "img:x.png"})  # no raise


def test_validate_labels_raises_on_unknown_id():
    labels = [Label(query="q1", expected=("code:a.py#f",))]
    with pytest.raises(ValueError, match="code:a.py#f"):
        validate_labels(labels, valid_ids={"tbl:data.csv"})


def test_validate_labels_error_mentions_the_offending_query():
    labels = [Label(query="what is the retry backoff", expected=("code:missing.py#f",))]
    with pytest.raises(ValueError, match="what is the retry backoff"):
        validate_labels(labels, valid_ids=set())


def test_validate_labels_aggregates_multiple_errors():
    labels = [
        Label(query="q1", expected=("code:missing1.py#f",)),
        Label(query="q2", expected=("code:missing2.py#g",)),
    ]
    with pytest.raises(ValueError) as excinfo:
        validate_labels(labels, valid_ids=set())
    message = str(excinfo.value)
    assert "code:missing1.py#f" in message
    assert "code:missing2.py#g" in message


def test_validate_labels_partial_hit_in_multi_id_expected_is_valid():
    # OR semantics belong to hit-rate scoring, not validation, as long as
    # every listed id is a real id in the index, validation passes even if
    # only one of them would ever actually be returned.
    labels = [Label(query="q1", expected=("tbl:data.csv", "pdf:bench.pdf#p3"))]
    validate_labels(labels, valid_ids={"tbl:data.csv", "pdf:bench.pdf#p3"})  # no raise
