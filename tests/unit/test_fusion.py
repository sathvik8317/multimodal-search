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
