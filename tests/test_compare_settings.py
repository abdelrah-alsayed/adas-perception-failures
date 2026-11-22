from explain_adas.compare_settings import BASE, COMPARISONS, compare_case


def test_compare_case_gives_one_row_per_comparison():
    same = list(range(30))
    flipped = list(reversed(range(30)))
    ks_case = {BASE: same,
               COMPARISONS["ks_seed_2"]: same,
               COMPARISONS["budget_1000"]: flipped,
               COMPARISONS["grey_mask"]: same}
    baseline_case = {"blur": same}
    rows = {r["comparison"]: r for r in compare_case(ks_case, baseline_case)}
    assert set(rows) == {"ks_seed_2", "budget_1000", "grey_mask", "hide_one_blur"}
    assert rows["ks_seed_2"]["spearman"] == 1.0
    assert rows["budget_1000"]["spearman"] == -1.0
    assert rows["budget_1000"]["top3_overlap"] == 0
    assert rows["hide_one_blur"]["top3_overlap"] == 3
    assert rows["ks_seed_2"]["top1_overlap"] == 1
    assert rows["budget_1000"]["top1_overlap"] == 0
    assert rows["ks_seed_2"]["top5_overlap"] == 5
