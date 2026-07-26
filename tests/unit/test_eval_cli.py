import json

from mmsearch.eval.cli import _MemoizingEmbedder, _print_compare, _print_sweep, build_arg_parser


def _report(hit_rate, fp_rate, per_query, ablation_hit_rates=None):
    report = {
        "aggregate_hit_rate": hit_rate,
        "false_positive_rate": fp_rate,
        "per_modality": {"code": hit_rate},
        "per_text_source": {"code_source": hit_rate},
        "per_query": per_query,
    }
    if ablation_hit_rates is not None:
        report["ablation_hit_rates"] = ablation_hit_rates
    return report


def test_print_compare_reports_fixed_and_regressed_queries(capsys):
    before = _report(
        0.5,
        0.4,
        [
            {"query": "a", "hit": True},
            {"query": "b", "hit": False},
            {"query": "neg", "negative": True, "false_positive": True},
        ],
    )
    after = _report(
        0.75,
        0.2,
        [
            {"query": "a", "hit": False},  # regressed
            {"query": "b", "hit": True},  # fixed
            {"query": "neg", "negative": True, "false_positive": False},  # fixed
        ],
    )

    _print_compare(before, after)
    out = capsys.readouterr().out

    assert "flipped queries (3)" in out
    assert "[REGRESSED] 'a'" in out
    assert "[FIXED] 'b'" in out
    assert "[FIXED] 'neg'" in out


def test_print_compare_no_flips_reports_zero(capsys):
    report = _report(0.8, 0.1, [{"query": "a", "hit": True}])
    _print_compare(report, json.loads(json.dumps(report)))
    out = capsys.readouterr().out
    assert "flipped queries (0)" in out


def test_print_sweep_lists_every_threshold(capsys):
    rows = [
        (0.05, {"aggregate_hit_rate": 0.76, "false_positive_rate": 0.4}),
        (0.10, {"aggregate_hit_rate": 0.76, "false_positive_rate": 0.2}),
    ]
    _print_sweep(rows)
    out = capsys.readouterr().out
    assert "0.05" in out and "0.10" in out
    assert "0.400" in out and "0.200" in out


def test_arg_parser_defaults_to_rrf_rerank_mode():
    args = build_arg_parser().parse_args([])
    assert args.mode == "rrf+rerank"
    assert args.ablations is False
    assert args.attribute is False
    assert args.compare is None


def test_arg_parser_compare_takes_two_paths():
    args = build_arg_parser().parse_args(["--compare", "a.json", "b.json"])
    assert args.compare == ["a.json", "b.json"]


def test_print_compare_shows_ablation_hit_rates_when_present(capsys):
    before = _report(
        0.5,
        0.4,
        [],
        ablation_hit_rates={
            "vector-only": {"aggregate_hit_rate": 0.333, "false_positive_rate": 1.0},
            "rrf-only": {"aggregate_hit_rate": 0.542, "false_positive_rate": 1.0},
            "rrf+rerank": {"aggregate_hit_rate": 0.917, "false_positive_rate": 0.2},
        },
    )
    after = _report(
        0.917,
        0.2,
        [],
        ablation_hit_rates={
            "vector-only": {"aggregate_hit_rate": 0.958, "false_positive_rate": 1.0},
            "rrf-only": {"aggregate_hit_rate": 0.958, "false_positive_rate": 1.0},
            "rrf+rerank": {"aggregate_hit_rate": 0.917, "false_positive_rate": 0.2},
        },
    )

    _print_compare(before, after)
    out = capsys.readouterr().out

    assert "ablation_hit_rates" in out
    assert "vector-only" in out and "rrf-only" in out and "rrf+rerank" in out
    assert "0.333" in out and "0.958" in out


def test_print_compare_omits_ablation_section_when_absent_from_both(capsys):
    # Older saved reports (e.g. eval_runs/baseline.json from before this key
    # existed) must still --compare cleanly.
    before = _report(0.8, 0.1, [{"query": "a", "hit": True}])
    after = _report(0.8, 0.1, [{"query": "a", "hit": True}])

    _print_compare(before, after)
    out = capsys.readouterr().out

    assert "ablation_hit_rates" not in out


def test_memoizing_embedder_calls_inner_once_per_distinct_query():
    calls = []

    class _CountingEmbedder:
        def embed_query(self, text):
            calls.append(text)
            return [float(len(text))]

        def embed_documents(self, items):
            raise AssertionError("eval CLI never calls embed_documents")

    wrapped = _MemoizingEmbedder(_CountingEmbedder())

    first = wrapped.embed_query("hello")
    second = wrapped.embed_query("hello")
    third = wrapped.embed_query("longer query")

    assert first == second == [5.0]
    assert third == [12.0]
    assert calls == ["hello", "longer query"]  # "hello" queried once despite two calls
