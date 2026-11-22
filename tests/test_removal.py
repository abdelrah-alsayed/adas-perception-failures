import numpy as np
import pytest

from explain_adas.removal import case_summary, curve_area, make_orders, removal_masks


def test_removal_masks_hide_one_more_region_in_each_row():
    masks = removal_masks([2, 0, 1], n_regions=3)
    assert masks.shape == (4, 3)
    assert masks[0].tolist() == [1, 1, 1]   # nothing hidden
    assert masks[1].tolist() == [1, 1, 0]   # region 2 hidden
    assert masks[2].tolist() == [0, 1, 0]   # regions 2 and 0 hidden
    assert masks[3].tolist() == [0, 0, 0]   # everything hidden


def test_make_orders_sorts_by_value_and_random_orders_are_permutations():
    orders = make_orders([0.1, 0.5, -0.2, 0.3], n_random=5, seed=42)
    assert orders["top_down"] == [1, 3, 0, 2]
    assert orders["bottom_up"] == [2, 0, 3, 1]
    random_names = [name for name in orders if name.startswith("random_")]
    assert len(random_names) == 5
    for name in random_names:
        assert sorted(orders[name]) == [0, 1, 2, 3]


def test_make_orders_same_seed_same_random_orders():
    a = make_orders([0.1, 0.5, -0.2, 0.3], n_random=3, seed=7)
    b = make_orders([0.1, 0.5, -0.2, 0.3], n_random=3, seed=7)
    assert a == b


def test_curve_area_is_lower_when_the_score_falls_faster():
    fast = curve_area([1.0, 0.0, 0.0], f_full=1.0)
    slow = curve_area([1.0, 1.0, 0.0], f_full=1.0)
    assert fast == pytest.approx(1 / 3)
    assert slow == pytest.approx(2 / 3)


def test_case_summary_counts_random_orders_that_are_beaten():
    curves = {"top_down": [1.0, 0.0, 0.0],      # falls fastest
              "bottom_up": [1.0, 1.0, 1.0],     # never falls
              "random_0": [1.0, 0.5, 0.0],
              "random_1": [1.0, 0.8, 0.0]}
    summary = case_summary(curves, f_full=1.0)
    assert summary["top_down_beats"] == 2       # both random curves fall slower
    assert summary["bottom_up_beats"] == 2      # both random curves fall faster
    assert summary["n_random"] == 2
    assert summary["bottom_up_peak"] == pytest.approx(1.0)
    assert np.isfinite(summary["auc_random_mean"])
