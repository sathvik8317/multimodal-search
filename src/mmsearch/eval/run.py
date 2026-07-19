"""Hit-rate@k evaluation runner (see PLAN.md §5).

Scoring semantics (definition of record):

    A query's ``expected`` ids are an OR-set of acceptable answers: a hit
    if ANY of them appears in the top-k returned ids. Each query contributes
    exactly 0 or 1 to the aggregate hit rate (never fractional, never AND).

Per-modality / per-text_source breakdowns use "listed vs. hit" attribution:
every modality/text_source referenced by a query's ``expected`` ids gets its
denominator incremented regardless of whether the query hit or missed; only
the modality/text_source of an id that actually appeared in the *hit set*
gets its numerator incremented. See ``evaluate()`` for the worked example.
"""

from __future__ import annotations

from dataclasses import dataclass

from mmsearch import config
from mmsearch.eval.dataset import Label
from mmsearch.retrieve.types import SearchFn
from mmsearch.schema import Modality, TextSource


def score_hit(expected: tuple[str, ...], returned_ids: list[str], k: int) -> bool:
    return bool(set(expected) & set(returned_ids[:k]))


@dataclass(frozen=True)
class EvalReport:
    aggregate_hit_rate: float
    per_modality: dict[str, float]
    per_text_source: dict[str, float]


def evaluate(
    search_fn: SearchFn,
    labels: list[Label],
    id_index: dict[str, tuple[Modality, TextSource]],
    k: int = config.TOP_K,
) -> EvalReport:
    aggregate_hits = 0
    per_modality_num: dict[str, int] = {}
    per_modality_den: dict[str, int] = {}
    per_text_source_num: dict[str, int] = {}
    per_text_source_den: dict[str, int] = {}

    for label in labels:
        returned_ids = [r.id for r in search_fn(label.query, k)]
        hit_ids = set(label.expected) & set(returned_ids[:k])

        modalities_listed = {id_index[eid][0].value for eid in label.expected}
        text_sources_listed = {id_index[eid][1].value for eid in label.expected}

        if hit_ids:
            aggregate_hits += 1
            hit_modalities = {id_index[eid][0].value for eid in hit_ids}
            hit_text_sources = {id_index[eid][1].value for eid in hit_ids}
        else:
            hit_modalities = set()
            hit_text_sources = set()

        for m in modalities_listed:
            per_modality_den[m] = per_modality_den.get(m, 0) + 1
            if m in hit_modalities:
                per_modality_num[m] = per_modality_num.get(m, 0) + 1

        for t in text_sources_listed:
            per_text_source_den[t] = per_text_source_den.get(t, 0) + 1
            if t in hit_text_sources:
                per_text_source_num[t] = per_text_source_num.get(t, 0) + 1

    aggregate_hit_rate = aggregate_hits / len(labels)
    per_modality = {m: per_modality_num.get(m, 0) / den for m, den in per_modality_den.items()}
    per_text_source = {
        t: per_text_source_num.get(t, 0) / den for t, den in per_text_source_den.items()
    }

    return EvalReport(
        aggregate_hit_rate=aggregate_hit_rate,
        per_modality=per_modality,
        per_text_source=per_text_source,
    )


def run_ablations(
    search_fns: dict[str, SearchFn],
    labels: list[Label],
    id_index: dict[str, tuple[Modality, TextSource]],
    k: int = config.TOP_K,
) -> dict[str, EvalReport]:
    return {mode: evaluate(fn, labels, id_index, k) for mode, fn in search_fns.items()}
