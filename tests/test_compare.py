from explain_adas.compare import rank_correlation, top_k_overlap


def test_same_order_gives_one():
    assert rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0


def test_opposite_order_gives_minus_one():
    assert rank_correlation([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_top3_overlap_counts_shared_regions():
    a = [5, 4, 3, 2, 1]   # top 3 regions: 0, 1, 2
    b = [5, 4, 1, 2, 3]   # top 3 regions: 0, 1, 4
    assert top_k_overlap(a, b, k=3) == 2
