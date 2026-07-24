"""Assembled search pipeline: vector + FTS retrieval, RRF fusion, optional rerank.

Scoring convention (documented since Phase-0 doesn't dictate one):
  - When results come from the reranker, `SearchResult.score` is the
    reranker's `relevance_score` directly.
  - Otherwise (vector-only, rrf-only, or any RRF fallback path),
    `SearchResult.score` is a positional score `1 / (rank + 1)` over the
    final returned order, so scores are always descending and comparable
    within a single response, even though they aren't comparable across
    modes or across separate calls.

`min_score_threshold` (config.MIN_SCORE_THRESHOLD, default 0.0) is applied
uniformly across both scoring regimes above -- see build_search_fn's
docstring for what that means in practice for each.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException

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


def _above_threshold(results: list[SearchResult], threshold: float) -> list[SearchResult]:
    """Drop any result scoring below threshold -- low-confidence noise is
    worse than an honest empty list. `>=`, not `>`: with the default 0.0,
    every real score (Cohere relevance_score, and the RRF-fallback
    1/(rank+1)) is non-negative, so nothing is filtered unless a caller
    explicitly raises the threshold above 0."""
    return [result for result in results if result.score >= threshold]


def _safe_embed_query(embedder: EmbeddingClient, query: str, provider: str) -> list[float] | None:
    """embed_query, but a provider outage must never surface as a 500 -- same
    graceful-degradation policy as the reranker fallback below: drop this
    retriever from RRF instead of failing the whole request."""
    try:
        return embedder.embed_query(query)
    except Exception:
        logger.warning(
            "%s embedder.embed_query failed; dropping it from retrieval", provider, exc_info=True
        )
        return None


def build_search_fn(
    table,
    cohere_embedder: EmbeddingClient,
    openai_embedder: EmbeddingClient,
    reranker: Reranker | None,
    *,
    mode: str = "rrf+rerank",
    fetch_n: int = config.FETCH_N,
    rerank_m: int = config.RERANK_M,
    rrf_k: int = config.RRF_K,
    min_score_threshold: float = config.MIN_SCORE_THRESHOLD,
) -> SearchFn:
    """Two vector retrievers (Cohere image-space, OpenAI text-space) + FTS,
    fused by three-way RRF, optionally reranked. See
    EMBEDDING_MIGRATION_PLAN.md: a query vector from one provider is only ever
    compared against that same provider's column -- never mixed.

    min_score_threshold filters the final result list before it's returned:
    a result scoring below it is dropped, and if nothing clears it the
    response is an empty list rather than low-confidence noise. Because the
    two scoring regimes mean different things (see module docstring), the
    same threshold value behaves differently by mode: in rerank mode it's a
    genuine relevance-quality filter; in vector-only/rrf-only/a
    failed-or-absent reranker, scores are positional (1/(rank+1)), so a
    nonzero threshold acts as an effective top-N position cap instead.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"unknown mode: {mode!r}; expected one of {_VALID_MODES}")

    def search(query: str, k: int = config.TOP_K) -> list[SearchResult]:
        # ponytail: sequential query embeds (2 network calls); wrap in
        # ThreadPool(2) if search latency matters.
        cohere_query_vector = _safe_embed_query(cohere_embedder, query, "cohere")
        openai_query_vector = _safe_embed_query(openai_embedder, query, "openai")

        if mode == "vector-only" and cohere_query_vector is None and openai_query_vector is None:
            # No FTS fallback in this mode -- an empty result list here would
            # look like "no matches" instead of "the search backend is down".
            raise HTTPException(
                status_code=503, detail="search unavailable: both embedders failed"
            )

        cohere_hits = (
            table.search(cohere_query_vector, query_type="vector", vector_column_name="vector_cohere")
            .limit(fetch_n)
            .to_list()
            if cohere_query_vector is not None
            else []
        )
        openai_hits = (
            table.search(openai_query_vector, query_type="vector", vector_column_name="vector_openai")
            .limit(fetch_n)
            .to_list()
            if openai_query_vector is not None
            else []
        )

        id_to_row = {row["id"]: row for row in cohere_hits}
        id_to_row.update({row["id"]: row for row in openai_hits})

        cohere_ids = [row["id"] for row in cohere_hits]
        openai_ids = [row["id"] for row in openai_hits]

        if mode == "vector-only":
            fused_ids = reciprocal_rank_fusion([cohere_ids, openai_ids], k=rrf_k)
            return _above_threshold(_positional_results(fused_ids[:k], id_to_row), min_score_threshold)

        fts_hits = table.search(query, query_type="fts").limit(fetch_n).to_list()
        id_to_row.update({row["id"]: row for row in fts_hits})
        fts_ids = [row["id"] for row in fts_hits]

        fused_ids = reciprocal_rank_fusion([cohere_ids, openai_ids, fts_ids], k=rrf_k)

        if mode == "rrf-only":
            return _above_threshold(_positional_results(fused_ids[:k], id_to_row), min_score_threshold)

        # mode == "rrf+rerank"
        if reranker is None:
            return _above_threshold(_positional_results(fused_ids[:k], id_to_row), min_score_threshold)

        shortlist_ids = fused_ids[:rerank_m]
        shortlist_docs = [id_to_row[id_]["content_text"] for id_ in shortlist_ids]
        try:
            rerank_results = reranker.rerank(query, shortlist_docs, top_n=k)
        except Exception:
            logger.warning(
                "reranker.rerank failed; falling back to RRF-fused order", exc_info=True
            )
            return _above_threshold(_positional_results(fused_ids[:k], id_to_row), min_score_threshold)

        reranked_results = [
            _row_to_result(id_to_row[shortlist_ids[rr.index]], score=rr.relevance_score)
            for rr in rerank_results
        ]
        return _above_threshold(reranked_results, min_score_threshold)

    return search
