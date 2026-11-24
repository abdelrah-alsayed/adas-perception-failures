import numpy as np

from explain_adas.smoke_test import check_values


def good_values():
    values = np.full(30, 0.01)          # sums to 0.30
    return 0.30, 0.0, values


def test_check_values_all_pass_for_a_consistent_case():
    f_full, f_empty, values = good_values()
    checks = check_values(0.30, f_full, f_empty, values, values)
    assert all(ok for _, ok, _ in checks)


def test_check_values_fail_when_values_do_not_add_up():
    f_full, f_empty, values = good_values()
    checks = dict((name, ok) for name, ok, _ in check_values(0.30, f_full, f_empty, values * 2, values * 2))
    assert checks["values add up to f_full - f_empty"] is False


def test_check_values_fail_when_the_stored_score_differs():
    f_full, f_empty, values = good_values()
    checks = dict((name, ok) for name, ok, _ in check_values(0.50, f_full, f_empty, values, values))
    assert checks["f_full equals the score stored in splits/cases.csv"] is False


def test_check_values_fail_when_two_runs_differ():
    f_full, f_empty, values = good_values()
    checks = dict((name, ok) for name, ok, _ in check_values(0.30, f_full, f_empty, values, values + 0.1))
    assert checks["same seed gives the same values (second run)"] is False
