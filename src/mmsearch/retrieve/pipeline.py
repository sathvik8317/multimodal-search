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

import json
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
_RERANK_SUMMARY_MAX_CHARS = 500


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


def _table_rerank_summary(row: dict) -> str:
    """A short synthetic description of a table row, for the reranker only.

    Diagnostic finding: the full markdown table blob (up to 12KB,
    config.MAX_TABLE_EMBED_CHARS) sent as-is scored 0.39-0.66 on completely
    unrelated queries -- Cohere Rerank has no natural-language signal to
    judge a huge pipe-delimited grid against and appears to default toward a
    moderate-to-high score regardless of topic. This summary gives it an
    actual sentence to compare against the query instead.
    """
    metadata = json.loads(row["metadata"])
    filename = row["source_path"].rsplit("/", 1)[-1]
    columns = metadata.get("columns", [])
    total_rows = metadata.get("total_rows", metadata.get("n_rows", "unknown"))
    sample_lines = row["content_text"].split("\n")[2:4]  # skip header + separator
    summary = (
        f"Table: {filename}. Columns: {', '.join(columns)}. "
        f"{total_rows} rows. Sample: {' '.join(sample_lines)}"
    )
    return summary[:_RERANK_SUMMARY_MAX_CHARS]


def _rerank_text(row: dict) -> str:
    """Text sent to the reranker for a candidate row. Embedding and FTS
    always use content_text unchanged (see ingest/tables.py, db.py) -- this
    exists solely to fix table rows' rerank input; every other modality is a
    passthrough.
    """
    if Modality(row["modality"]) is Modality.TABLE:
        return _table_rerank_summary(row)
    return row["content_text"]


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


def _build_eligibility(table) -> tuple[set[str], set[str]]:
    """Static, computed once per build_search_fn call (table load), not per
    query: which ids each vector retriever could ever structurally return,
    independent of any query's fetch_n truncation. Data-driven -- checks
    actual column nullness -- rather than hardcoding which modalities get
    which vectors, since that's a real per-row fact (see
    EMBEDDING_MIGRATION_PLAN.md) that fusion.py has no business assuming.

    Needed to fix a real bug: a pdf_page/diagram row can appear in both the
    Cohere and OpenAI vector lists, while a code/table row only ever has an
    OpenAI vector and can appear in one -- reciprocal_rank_fusion's plain sum
    rewards that extra list presence regardless of rank. eligible_universes
    normalizes it away (see fusion.py). FTS has no such restriction --
    content_text is never empty (Row.__post_init__) -- so it needs no
    eligibility set; every id is always eligible for it.
    """
    rows = table.search().select(["id", "vector_cohere", "vector_openai"]).to_list()
    cohere_eligible = {row["id"] for row in rows if row["vector_cohere"] is not None}
    openai_eligible = {row["id"] for row in rows if row["vector_openai"] is not None}
    return cohere_eligible, openai_eligible


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

    cohere_eligible_ids, openai_eligible_ids = _build_eligibility(table)

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

        # A retriever that failed this query (embed_query returned None) must
        # not count toward any id's eligibility denominator either -- it
        # never got a chance to score anything this time, so treat it as
        # absent entirely rather than unfairly halving a row that's
        # structurally eligible for it (see fusion.py's eligible_universes
        # docstring).
        cohere_universe_this_query = cohere_eligible_ids if cohere_query_vector is not None else set()
        openai_universe_this_query = openai_eligible_ids if openai_query_vector is not None else set()

        if mode == "vector-only":
            fused_ids = reciprocal_rank_fusion(
                [cohere_ids, openai_ids],
                k=rrf_k,
                eligible_universes=[cohere_universe_this_query, openai_universe_this_query],
            )
            return _above_threshold(_positional_results(fused_ids[:k], id_to_row), min_score_threshold)

        fts_hits = table.search(query, query_type="fts").limit(fetch_n).to_list()
        id_to_row.update({row["id"]: row for row in fts_hits})
        fts_ids = [row["id"] for row in fts_hits]

        # FTS has no structural eligibility restriction -- content_text is
        # never empty, so every row is always FTS-eligible. fusion.py only
        # ever checks eligibility for an id that has a score, i.e. one that
        # already appears in cohere_ids/openai_ids/fts_ids, so "every id
        # across those three lists" is equivalent to "every id" for this
        # purpose -- no separate full-table scan needed.
        all_ids = set(cohere_ids) | set(openai_ids) | set(fts_ids)
        fused_ids = reciprocal_rank_fusion(
            [cohere_ids, openai_ids, fts_ids],
            k=rrf_k,
            eligible_universes=[cohere_universe_this_query, openai_universe_this_query, all_ids],
        )

        if mode == "rrf-only":
            return _above_threshold(_positional_results(fused_ids[:k], id_to_row), min_score_threshold)

        # mode == "rrf+rerank"
        if reranker is None:
            return _above_threshold(_positional_results(fused_ids[:k], id_to_row), min_score_threshold)

        shortlist_ids = fused_ids[:rerank_m]
        shortlist_docs = [_rerank_text(id_to_row[id_]) for id_ in shortlist_ids]
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
