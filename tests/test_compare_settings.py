from explain_adas.compare import top_k_overlap
from explain_adas.compare_settings import (BASE, COMPARISONS, compare_case, object_region,
                                          top_k_overlap_without)


def test_compare_case_gives_one_row_per_comparison():
    same = list(range(30))
    flipped = list(reversed(range(30)))
    ks_case = {BASE: same,
               COMPARISONS["ks_seed_2"]: same,
               COMPARISONS["budget_1000"]: flipped,
               COMPARISONS["grey_mask"]: same}
    baseline_case = {"blur": same}
    rows = {r["comparison"]: r for r in compare_case(ks_case, baseline_case, skip=0)}
    assert set(rows) == {"ks_seed_2", "budget_1000", "grey_mask", "hide_one_blur"}
    assert rows["ks_seed_2"]["spearman"] == 1.0
    assert rows["budget_1000"]["spearman"] == -1.0
    assert rows["budget_1000"]["top3_overlap"] == 0
    assert rows["hide_one_blur"]["top3_overlap"] == 3
    assert rows["ks_seed_2"]["top1_overlap"] == 1
    assert rows["budget_1000"]["top1_overlap"] == 0
    assert rows["ks_seed_2"]["top5_overlap"] == 5


def test_overlap_without_skips_the_object_region():
    a = [9, 8, 7, 1, 0]   # top 3: regions 0, 1, 2
    b = [9, 1, 0, 8, 7]   # top 3: regions 0, 3, 4
    assert top_k_overlap(a, b, k=3) == 1
    # without region 0: a gives {1, 2, 3}, b gives {3, 4, 1}
    assert top_k_overlap_without(a, b, skip=0, k=3) == 2
    assert a[0] == 9      # the input list is not changed


def test_object_region_uses_the_box_centre():
    boxes = [(0, 0, 10, 10), (10, 0, 20, 10)]
    assert object_region([12, 2, 16, 6], boxes) == 1
    assert object_region([8, 2, 12, 6], boxes) == 1   # centre x = 10 is in the second box
