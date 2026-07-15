"""Reciprocal Rank Fusion for combining multiple ranked id lists."""

from __future__ import annotations

from mmsearch import config


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = config.RRF_K) -> list[str]:
    """Fuse multiple ranked lists of ids into a single ranked list.

    For each id, score = sum(1 / (k + rank)) across every list in which it
    appears (0-indexed rank within that list; absent from a list contributes
    0 for that list). Results are sorted by descending fused score, with a
    deterministic tiebreak on the id string itself.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, id_ in enumerate(ranked_list):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)

    return sorted(scores.keys(), key=lambda id_: (-scores[id_], id_))
