"""Assembled search pipeline: vector + FTS retrieval, RRF fusion, optional rerank.

Scoring convention (documented since Phase-0 doesn't dictate one):
  - When results come from the reranker, `SearchResult.score` is the
    reranker's `relevance_score` directly.
  - Otherwise (vector-only, rrf-only, or any RRF fallback path),
    `SearchResult.score` is a positional score `1 / (rank + 1)` over the
    final returned order, so scores are always descending and comparable
    within a single response, even though they aren't comparable across
    modes or across separate calls.
"""

from __future__ import annotations

import logging

from mmsearch import config
from mmsearch.clients.protocols import EmbeddingClient, Reranker
from mmsearch.retrieve.fusion import reciprocal_rank_fusion
from mmsearch.retrieve.types import SearchFn, SearchResult
from mmsearch.schema import Modality, TextSource

logger = logging.getLogger(__name__)

_SNIPPET_LEN = 200
_VALID_MODES = ("vector-only", "rrf-only", "rrf+rerank")


def _build_snippet(content_text: str, modality: Modality) -> str:
    """Modality-aware display snippet. A flat content_text[:_SNIPPET_LEN]
    slice works fine for prose (pdf_page, diagram) but is wrong for
    structured content:
      - table: a 200-char slice barely covers the header row, cutting the
        first data row mid-value. Show the header + separator + first 3
        complete data rows instead (split on newline, never mid-row).
      - code: content_text is prefixed with a "# file: ... / # language:
        ... [/ # class: ...]" context header (2 lines for top-level
        functions/classes, 3 for methods) that's deliberately embedded for
        retrieval quality but reads as noise in a result card. Skip past
        all leading '#' comment lines so the snippet starts at the actual
        def/class line, regardless of header length.
    """
    if modality is Modality.TABLE:
        lines = content_text.split("\n")
        return "\n".join(lines[:5])  # header + separator + up to 3 data rows

    if modality is Modality.CODE:
        lines = content_text.split("\n")
        start = 0
        while start < len(lines) and lines[start].lstrip().startswith("#"):
            start += 1
        return "\n".join(lines[start:])[:_SNIPPET_LEN]

    return content_text[:_SNIPPET_LEN]


def _row_to_result(row: dict, score: float) -> SearchResult:
    modality = Modality(row["modality"])
    return SearchResult(
        id=row["id"],
        modality=modality,
        score=score,
        snippet=_build_snippet(row["content_text"], modality),
        thumbnail_ref=row["thumbnail_ref"],
        source_path=row["source_path"],
        text_source=TextSource(row["text_source"]),
    )


def _positional_results(ids: list[str], id_to_row: dict[str, dict]) -> list[SearchResult]:
    return [
        _row_to_result(id_to_row[id_], score=1.0 / (rank + 1))
        for rank, id_ in enumerate(ids)
    ]


def build_search_fn(
    table,
    embedding_client: EmbeddingClient,
    reranker: Reranker | None,
    *,
    mode: str = "rrf+rerank",
    fetch_n: int = config.FETCH_N,
    rerank_m: int = config.RERANK_M,
    rrf_k: int = config.RRF_K,
) -> SearchFn:
    if mode not in _VALID_MODES:
        raise ValueError(f"unknown mode: {mode!r}; expected one of {_VALID_MODES}")

    def search(query: str, k: int = config.TOP_K) -> list[SearchResult]:
        query_vector = embedding_client.embed_query(query)
        vector_hits = table.search(query_vector, query_type="vector").limit(fetch_n).to_list()

        if mode == "vector-only":
            id_to_row = {row["id"]: row for row in vector_hits}
            vector_ids = [row["id"] for row in vector_hits][:k]
            return _positional_results(vector_ids, id_to_row)

        fts_hits = table.search(query, query_type="fts").limit(fetch_n).to_list()

        id_to_row = {row["id"]: row for row in vector_hits}
        id_to_row.update({row["id"]: row for row in fts_hits})

        vector_ids = [row["id"] for row in vector_hits]
        fts_ids = [row["id"] for row in fts_hits]
        fused_ids = reciprocal_rank_fusion([vector_ids, fts_ids], k=rrf_k)

        if mode == "rrf-only":
            return _positional_results(fused_ids[:k], id_to_row)

        # mode == "rrf+rerank"
        if reranker is None:
            return _positional_results(fused_ids[:k], id_to_row)

        shortlist_ids = fused_ids[:rerank_m]
        shortlist_docs = [id_to_row[id_]["content_text"] for id_ in shortlist_ids]
        try:
            rerank_results = reranker.rerank(query, shortlist_docs, top_n=k)
        except Exception:
            logger.warning(
                "reranker.rerank failed; falling back to RRF-fused order", exc_info=True
            )
            return _positional_results(fused_ids[:k], id_to_row)

        return [
            _row_to_result(id_to_row[shortlist_ids[rr.index]], score=rr.relevance_score)
            for rr in rerank_results
        ]

    return search
