"""Reciprocal Rank Fusion for combining multiple ranked id lists."""

from __future__ import annotations

from mmsearch import config


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = config.RRF_K,
    eligible_universes: list[set[str]] | None = None,
) -> list[str]:
    """Fuse multiple ranked lists of ids into a single ranked list.

    For each id, score = sum(1 / (k + rank)) across every list in which it
    appears (0-indexed rank within that list; absent from a list contributes
    0 for that list). Results are sorted by descending fused score, with a
    deterministic tiebreak on the id string itself.

    ``eligible_universes`` (optional, one set per entry in ``ranked_lists``)
    normalizes each id's summed score by how many lists it was *structurally
    eligible* for, not by how many lists it happened to appear in. Without
    it, an id present in more lists always out-scores one present in fewer,
    independent of rank -- e.g. a row with vectors in two retrievers beats a
    row that only ever has one, even when its individual ranks are worse on
    average. Each ``eligible_universes[i]`` should be "every id retriever i
    could ever return, before any fetch-depth truncation" -- pass an empty
    set for a retriever that failed/was skipped this query, so ids that are
    structurally eligible for it aren't divided by a list that never got a
    chance to score them. ``None`` (the default) preserves the plain-sum
    behavior above exactly.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, id_ in enumerate(ranked_list):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)

    if eligible_universes is None:
        return sorted(scores.keys(), key=lambda id_: (-scores[id_], id_))

    normalized = {
        id_: total / max(sum(1 for universe in eligible_universes if id_ in universe), 1)
        for id_, total in scores.items()
    }
    return sorted(normalized.keys(), key=lambda id_: (-normalized[id_], id_))
