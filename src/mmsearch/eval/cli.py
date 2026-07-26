"""`mmsearch-eval` entrypoint: run the harness, persist JSON, diff two runs.

PLAN.md §5.3 promised a `--mode` flag on a committed eval command; nothing
was ever built and every historical number in README.md/config.py's tuning
comments came from ad-hoc REPL calls to build_search_fn + run_ablations,
never persisted. This is that command. It is a thin driver around the
existing, unmodified eval/run.py and eval/dataset.py: no scoring logic
lives here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from lancedb.table import Table

from mmsearch import config, db
from mmsearch.eval.dataset import Label, load_labels, validate_labels
from mmsearch.eval.run import EvalReport, evaluate, false_positive_rate, run_ablations
from mmsearch.retrieve.pipeline import _VALID_MODES, build_search_fn
from mmsearch.retrieve.types import SearchFn
from mmsearch.schema import Modality, TextSource

DEFAULT_LABELS_PATH = Path(__file__).parent / "labels.yaml"


def _build_id_index(table: Table) -> dict[str, tuple[Modality, TextSource]]:
    rows = table.search().select(["id", "modality", "text_source"]).to_list()
    return {
        row["id"]: (Modality(row["modality"]), TextSource(row["text_source"]))
        for row in rows
    }


def _build_clients() -> tuple[object, object, object]:
    """Real Cohere + OpenAI clients. Imported lazily so unit tests / --compare
    never need API keys or network access (mirrors ingest/cli.py's pattern).

    A full eval run fires ~2 Cohere calls/query (embed_query + rerank) back
    to back with no pacing, well past a trial key's 10-calls/minute ceiling
    -- production's default max_retries=3/backoff_seconds=1.0 (~7s of total
    wait) isn't enough to clear that window, so a rate-limited query silently
    falls back to RRF-fused order (pipeline.py's existing degrade-on-error
    policy) instead of the real reranker, contaminating the very number this
    harness exists to measure. The eval CLI is a batch/offline context where
    latency doesn't matter and correctness does, so retry harder here instead
    of adding call-pacing logic: max_retries=6/backoff_seconds=2.0 gives
    2+4+8+16+32+64s of cumulative backoff, comfortably clearing a 60s trial
    window. Production's CohereClient() defaults are untouched.
    """
    from mmsearch.clients.cohere import CohereClient
    from mmsearch.clients.openai import OpenAIClient

    cohere_client = CohereClient(max_retries=6, backoff_seconds=2.0)
    openai_client = OpenAIClient()
    return cohere_client, openai_client, cohere_client  # cohere doubles as reranker


def _memoize_search(search_fn: SearchFn) -> SearchFn:
    """Cache by query text so evaluate()/false_positive_rate() and the
    per-query dump below never issue the same query twice -- important both
    for API cost and because a rate-limited call can silently degrade to
    RRF-fallback order (see _build_clients' docstring), so calling the same
    query twice could legitimately return two different answers."""
    cache: dict[str, list] = {}

    def wrapped(query: str, k: int = config.TOP_K) -> list:
        if query not in cache:
            cache[query] = search_fn(query, k=k)
        return cache[query]

    return wrapped


def _run_report(
    *,
    mode: str,
    threshold: float,
    labels_path: Path,
) -> dict:
    labels = load_labels(labels_path)
    table = db.open_table()
    id_index = _build_id_index(table)
    validate_labels(labels, valid_ids=set(id_index))

    cohere_client, openai_client, reranker = _build_clients()
    search_fn = _memoize_search(
        build_search_fn(
            table,
            cohere_client,
            openai_client,
            reranker,
            mode=mode,
            min_score_threshold=threshold,
        )
    )

    positive_labels = [label for label in labels if not label.negative]
    negative_labels = [label for label in labels if label.negative]

    report = evaluate(search_fn, positive_labels, id_index)
    fp_rate = false_positive_rate(search_fn, negative_labels)

    # Every label was already queried exactly once above; this replays the
    # memoized cache, issuing zero further search_fn calls.
    per_query = []
    for label in positive_labels:
        results = search_fn(label.query, k=config.TOP_K)
        returned_ids = [r.id for r in results]
        hit = bool(set(label.expected) & set(returned_ids))
        per_query.append(
            {
                "query": label.query,
                "expected": list(label.expected),
                "returned_ids": returned_ids,
                "scores": [r.score for r in results],
                "hit": hit,
            }
        )
    for label in negative_labels:
        results = search_fn(label.query, k=config.TOP_K)
        returned_ids = [r.id for r in results]
        per_query.append(
            {
                "query": label.query,
                "expected": [],
                "negative": True,
                "returned_ids": returned_ids,
                "scores": [r.score for r in results],
                "false_positive": bool(returned_ids),
            }
        )

    return {
        "mode": mode,
        "threshold": threshold,
        "aggregate_hit_rate": report.aggregate_hit_rate,
        "per_modality": report.per_modality,
        "per_text_source": report.per_text_source,
        "false_positive_rate": fp_rate,
        "per_query": per_query,
    }


def _run_ablations_report(*, threshold: float, labels_path: Path) -> dict:
    labels = load_labels(labels_path)
    table = db.open_table()
    id_index = _build_id_index(table)
    validate_labels(labels, valid_ids=set(id_index))

    cohere_client, openai_client, reranker = _build_clients()
    positive_labels = [label for label in labels if not label.negative]
    negative_labels = [label for label in labels if label.negative]

    search_fns: dict[str, SearchFn] = {
        mode: build_search_fn(
            table, cohere_client, openai_client, reranker, mode=mode, min_score_threshold=threshold
        )
        for mode in _VALID_MODES
    }
    reports: dict[str, EvalReport] = run_ablations(search_fns, positive_labels, id_index)

    return {
        mode: {
            "aggregate_hit_rate": report.aggregate_hit_rate,
            "per_modality": report.per_modality,
            "per_text_source": report.per_text_source,
            "false_positive_rate": false_positive_rate(search_fns[mode], negative_labels),
        }
        for mode, report in reports.items()
    }


def _attribution_report(*, labels_path: Path) -> dict:
    """Per positive label, per retriever: rank (0-indexed) of the first
    expected id in that retriever's raw fetch_n list, or None if absent.
    Diagnostic only -- not a production mode, does not touch _VALID_MODES."""
    labels = load_labels(labels_path)
    table = db.open_table()
    id_index = _build_id_index(table)
    validate_labels(labels, valid_ids=set(id_index))

    cohere_client, openai_client, _ = _build_clients()
    positive_labels = [label for label in labels if not label.negative]

    results = []
    for label in positive_labels:
        cohere_vec = cohere_client.embed_query(label.query)
        openai_vec = openai_client.embed_query(label.query)

        cohere_ids = [
            r["id"]
            for r in table.search(cohere_vec, query_type="vector", vector_column_name="vector_cohere")
            .limit(config.FETCH_N)
            .to_list()
        ]
        openai_ids = [
            r["id"]
            for r in table.search(openai_vec, query_type="vector", vector_column_name="vector_openai")
            .limit(config.FETCH_N)
            .to_list()
        ]
        fts_ids = [r["id"] for r in table.search(label.query, query_type="fts").limit(config.FETCH_N).to_list()]

        def rank_of(ids: list[str], expected: tuple[str, ...]) -> int | None:
            for rank, id_ in enumerate(ids):
                if id_ in expected:
                    return rank
            return None

        results.append(
            {
                "query": label.query,
                "expected": list(label.expected),
                "rank_cohere": rank_of(cohere_ids, label.expected),
                "rank_openai": rank_of(openai_ids, label.expected),
                "rank_fts": rank_of(fts_ids, label.expected),
            }
        )
    return {"attribution": results}


def _print_compare(before: dict, after: dict) -> None:
    print(f"{'metric':<30} {'before':>10} {'after':>10}")
    print(f"{'aggregate_hit_rate':<30} {before['aggregate_hit_rate']:>10.3f} {after['aggregate_hit_rate']:>10.3f}")
    print(f"{'false_positive_rate':<30} {before['false_positive_rate']:>10.3f} {after['false_positive_rate']:>10.3f}")

    modalities = sorted(set(before["per_modality"]) | set(after["per_modality"]))
    print("\nper_modality:")
    for m in modalities:
        b = before["per_modality"].get(m)
        a = after["per_modality"].get(m)
        print(f"  {m:<26} {b if b is not None else '-':>10} {a if a is not None else '-':>10}")

    sources = sorted(set(before["per_text_source"]) | set(after["per_text_source"]))
    print("\nper_text_source:")
    for t in sources:
        b = before["per_text_source"].get(t)
        a = after["per_text_source"].get(t)
        print(f"  {t:<26} {b if b is not None else '-':>10} {a if a is not None else '-':>10}")

    before_by_query = {q["query"]: q for q in before.get("per_query", [])}
    after_by_query = {q["query"]: q for q in after.get("per_query", [])}
    flipped = []
    for query, b in before_by_query.items():
        a = after_by_query.get(query)
        if a is None:
            continue
        b_ok = b.get("hit", not b.get("false_positive", False))
        a_ok = a.get("hit", not a.get("false_positive", False))
        if b_ok != a_ok:
            flipped.append((query, b_ok, a_ok))

    print(f"\nflipped queries ({len(flipped)}):")
    for query, b_ok, a_ok in flipped:
        direction = "FIXED" if a_ok and not b_ok else "REGRESSED"
        print(f"  [{direction}] {query!r}: {b_ok} -> {a_ok}")


def _print_sweep(rows: list[tuple[float, dict]]) -> None:
    print(f"{'threshold':>10} {'hit_rate@5':>12} {'false_positive_rate':>22}")
    for threshold, report in rows:
        print(f"{threshold:>10.2f} {report['aggregate_hit_rate']:>12.3f} {report['false_positive_rate']:>22.3f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmsearch-eval")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--mode", choices=_VALID_MODES, default="rrf+rerank")
    parser.add_argument("--threshold", type=float, default=config.MIN_SCORE_THRESHOLD)
    parser.add_argument("--out", type=Path, help="Write the JSON report here")
    parser.add_argument("--ablations", action="store_true", help="Run all three modes instead of --mode")
    parser.add_argument("--attribute", action="store_true", help="Per-retriever rank diagnostic (see Phase 2)")
    parser.add_argument(
        "--sweep-threshold",
        type=str,
        help="Comma-separated thresholds to sweep in --mode, e.g. 0.05,0.10,0.20,0.30",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE_JSON", "AFTER_JSON"),
        help="Diff two previously-saved --out reports instead of running anything",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.compare:
        before = json.loads(Path(args.compare[0]).read_text())
        after = json.loads(Path(args.compare[1]).read_text())
        _print_compare(before, after)
        return 0

    if args.attribute:
        report = _attribution_report(labels_path=args.labels)
        print(json.dumps(report, indent=2))
    elif args.ablations:
        report = _run_ablations_report(threshold=args.threshold, labels_path=args.labels)
        print(json.dumps(report, indent=2))
    elif args.sweep_threshold:
        thresholds = [float(t) for t in args.sweep_threshold.split(",")]
        rows = [
            (t, _run_report(mode=args.mode, threshold=t, labels_path=args.labels))
            for t in thresholds
        ]
        _print_sweep(rows)
        report = {"sweep": [{"threshold": t, **r} for t, r in rows]}
    else:
        report = _run_report(mode=args.mode, threshold=args.threshold, labels_path=args.labels)
        print(json.dumps({k: v for k, v in report.items() if k != "per_query"}, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
