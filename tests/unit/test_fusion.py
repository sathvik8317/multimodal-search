from mmsearch.retrieve.fusion import reciprocal_rank_fusion


def test_hand_computed_two_list_fusion_order():
    # k=1 for easy hand computation.
    # a: list1 rank0 -> 1/(1+0)=1.0                         total=1.0
    # b: list1 rank1 -> 1/(1+1)=0.5, list2 rank0 -> 1/(1+0)=1.0   total=1.5
    # c: list1 rank2 -> 1/(1+2)=0.3333, list2 rank1 -> 1/(1+1)=0.5 total=0.8333
    # d: list2 rank2 -> 1/(1+2)=0.3333                       total=0.3333
    ranked_lists = [["a", "b", "c"], ["b", "c", "d"]]

    result = reciprocal_rank_fusion(ranked_lists, k=1)

    assert result == ["b", "a", "c", "d"]


def test_single_list_is_order_preserving():
    ranked_lists = [["x", "y", "z"]]

    result = reciprocal_rank_fusion(ranked_lists, k=60)

    assert result == ["x", "y", "z"]


def test_ids_in_both_lists_rank_above_ids_in_one_list():
    # "a" appears in both lists at rank 0, "x" only in list1 at rank1,
    # "y" only in list2 at rank1. "a" should win; "x"/"y" tie and are
    # broken deterministically by id string.
    ranked_lists = [["a", "x"], ["a", "y"]]

    result = reciprocal_rank_fusion(ranked_lists, k=60)

    assert result == ["a", "x", "y"]


def test_empty_input_produces_empty_output():
    assert reciprocal_rank_fusion([]) == []


def test_list_of_empty_lists_produces_empty_output():
    assert reciprocal_rank_fusion([[], []]) == []


def test_default_k_uses_config_rrf_k():
    from mmsearch import config

    # With the default k, order should still follow RRF math for a simple case.
    ranked_lists = [["a", "b"], ["b", "a"]]

    result = reciprocal_rank_fusion(ranked_lists)

    # a: rank0 in list1 -> 1/(k+0), rank1 in list2 -> 1/(k+1)
    # b: rank1 in list1 -> 1/(k+1), rank0 in list2 -> 1/(k+0)
    # symmetric -> equal scores -> tiebreak alphabetically
    k = config.RRF_K
    expected_score_a = 1 / (k + 0) + 1 / (k + 1)
    expected_score_b = 1 / (k + 1) + 1 / (k + 0)
    assert expected_score_a == expected_score_b
    assert result == ["a", "b"]


# --- eligible_universes: normalize by structural eligibility, not raw list count -------------
#
# Bug this fixes: an id present in more lists always out-scores one present
# in fewer, *independent of rank*, even when the extra list presence isn't
# earned -- e.g. a pdf_page/diagram row is eligible for both the Cohere and
# OpenAI vector lists, while a code/table row only ever has an OpenAI
# vector and can never appear in the Cohere list. Traced concretely: the
# correct code row for "the make_id function..." ranked #1 in the
# OpenAI-only list but dropped to rank 17 after fusion, beaten by pdf_page
# rows present in both lists at worse individual ranks. eligible_universes
# normalizes each id's summed score by how many lists it was structurally
# eligible for this query (not how many it happened to appear in), so
# presence in more lists only helps when it's earned by rank, not merely by
# having more retrieval channels.


def test_no_eligible_universes_preserves_exact_current_sum_behavior():
    ranked_lists = [["a", "b", "c"], ["b", "c", "d"]]

    with_default = reciprocal_rank_fusion(ranked_lists, k=1)
    with_explicit_none = reciprocal_rank_fusion(ranked_lists, k=1, eligible_universes=None)

    assert with_default == with_explicit_none == ["b", "a", "c", "d"]


def test_single_eligible_list_id_score_is_unaffected_by_normalization():
    # "a" is eligible for only list1 -- dividing its score by 1 must be a
    # true no-op, identical to plain sum-RRF for the same input.
    ranked_lists = [["a"]]
    eligible_universes = [{"a"}]

    normalized = reciprocal_rank_fusion(ranked_lists, k=1, eligible_universes=eligible_universes)
    plain_sum = reciprocal_rank_fusion(ranked_lists, k=1)

    assert normalized == plain_sum == ["a"]


def test_multi_eligible_list_id_is_normalized_relative_to_single_eligible_id():
    # k=1. "a" only eligible for list1, ranks 0 there -> raw score 1.0.
    # "b" eligible for both lists: list1 rank1 -> 0.5, list2 rank0 -> 1.0,
    # raw sum 1.5.
    #
    # Plain sum-RRF ranks b above a (1.5 > 1.0) purely because b appears in
    # both lists -- this is the exact bug: b's *average* rank per list
    # (0.75) is actually worse than a's single-list rank (1.0).
    #
    # Normalized: a = 1.0/1 = 1.0 (only ever eligible for 1 list).
    #             b = 1.5/2 = 0.75 (eligible for 2 lists, divided by 2).
    # a now correctly outranks b.
    ranked_lists = [["a", "b"], ["b"]]
    eligible_universes = [{"a", "b"}, {"b"}]

    assert reciprocal_rank_fusion(ranked_lists, k=1) == ["b", "a"]

    normalized = reciprocal_rank_fusion(ranked_lists, k=1, eligible_universes=eligible_universes)
    assert normalized == ["a", "b"]


def test_failed_retriever_list_is_excluded_from_the_denominator():
    # A retriever that failed this query (e.g. embed_query raised, see
    # pipeline.py's _safe_embed_query) contributes an empty ranked list --
    # its eligible_universe for THIS query must also be empty, so ids that
    # are structurally eligible for it (e.g. a pdf_page row eligible for
    # both Cohere and OpenAI) aren't divided by a list that never got a
    # chance to score them. This must be indistinguishable from that list
    # never having existed for this query at all.
    with_failed_list = reciprocal_rank_fusion(
        [["a", "b"], []],
        k=1,
        eligible_universes=[{"a", "b"}, set()],
    )
    without_that_list_at_all = reciprocal_rank_fusion(
        [["a", "b"]],
        k=1,
        eligible_universes=[{"a", "b"}],
    )

    assert with_failed_list == without_that_list_at_all == ["a", "b"]
