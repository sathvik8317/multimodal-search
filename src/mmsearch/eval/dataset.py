"""labels.yaml format + loader + validator.

Format (see PLAN.md §5.1):

    - query: "the diagram showing auth token flow"
      expected: ["img:docs/auth-flow.png"]
    - query: "p99 latency numbers for the reranker"
      expected: ["tbl:data/latency.csv", "pdf:specs/bench.pdf#p3"]   # either is a hit (OR)

``expected`` is a set of acceptable answers combined with OR. See
mmsearch.eval.run for the hit-rate@5 scoring semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Label:
    query: str
    expected: tuple[str, ...]


def _build_label(entry: dict) -> Label:
    query = entry.get("query")
    if not isinstance(query, str) or not query:
        raise ValueError(f"label entry {entry!r} must have a non-empty string 'query'")

    expected = entry.get("expected")
    if not isinstance(expected, list) or not expected:
        raise ValueError(f"label for query {query!r} must have a non-empty 'expected' list")

    return Label(query=query, expected=tuple(expected))


def load_labels(path: str | Path) -> list[Label]:
    raw = yaml.safe_load(Path(path).read_text())
    if not raw:
        return []
    return [_build_label(entry) for entry in raw]


def validate_labels(labels: list[Label], valid_ids: set[str]) -> None:
    """Typo guard: raise if any expected id is absent from the index's id set."""
    errors = [
        f"query {label.query!r} references unknown id {id_!r}"
        for label in labels
        for id_ in label.expected
        if id_ not in valid_ids
    ]
    if errors:
        raise ValueError(
            "labels.yaml references ids not present in the index:\n" + "\n".join(errors)
        )
