import pytest

from explain_adas.compare_models import compare_models_case, is_flipped


def test_is_flipped_when_models_disagree():
    assert is_flipped([0.09, 0.53, 0.73])
    assert not is_flipped([0.60, 0.70, 0.30])   # all hits
    assert not is_flipped([0.01, 0.10, 0.24])   # all misses


def test_compare_models_case_compares_every_pair_of_models():
    same = [9, 8, 7, 1, 0]
    other = [9, 1, 0, 8, 7]
    maps = {"seed_0": same, "seed_1": same, "seed_2": other}
    rows = {r["models"]: r for r in compare_models_case(maps, skip=0)}
    assert set(rows) == {"seed_0 vs seed_1", "seed_0 vs seed_2", "seed_1 vs seed_2"}
    assert rows["seed_0 vs seed_1"]["spearman"] == pytest.approx(1.0)
    assert rows["seed_0 vs seed_1"]["top3_overlap"] == 3
    assert rows["seed_0 vs seed_2"]["top3_overlap"] == 1
    assert rows["seed_0 vs seed_2"]["top3_ctx_overlap"] == 2
