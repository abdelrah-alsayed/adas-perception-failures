import itertools
import math

import numpy as np
import shap

from explain_adas.kernel_shap import kernel_shap


def toy_f(masks):
    """Made-up model: two single regions, one pair that works together, one pair that fights."""
    m = np.asarray(masks)
    return (0.5 * m[:, 0] + 0.3 * m[:, 1] + 0.2 * m[:, 2] * m[:, 3]
            + 0.1 * m[:, 4] * (1 - m[:, 5]))


def exact_shapley(f, n):
    """The exact Shapley values, by trying every coalition (only for small n)."""
    values = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            for subset in itertools.combinations(others, size):
                without = np.zeros(n)
                without[list(subset)] = 1
                with_i = without.copy()
                with_i[i] = 1
                weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                values[i] += weight * (f(with_i[None, :])[0] - f(without[None, :])[0])
    return values


def test_exact_values_match_the_hand_calculation():
    expected = [0.5, 0.3, 0.1, 0.1, 0.05, -0.05, 0, 0]
    assert np.allclose(exact_shapley(toy_f, 8), expected)


def test_kernel_shap_with_all_coalitions_gives_the_exact_values():
    values = kernel_shap(toy_f, 8, nsamples=1000, seed=0)
    assert np.allclose(values, exact_shapley(toy_f, 8), atol=1e-6)


def test_values_add_up_to_full_minus_empty():
    values = kernel_shap(toy_f, 8, nsamples=1000, seed=0)
    full = toy_f(np.ones((1, 8)))[0]
    empty = toy_f(np.zeros((1, 8)))[0]
    assert abs(values.sum() - (full - empty)) < 1e-6


def test_same_seed_same_values_other_seed_other_values():
    a = kernel_shap(toy_f, 12, nsamples=200, seed=1)
    b = kernel_shap(toy_f, 12, nsamples=200, seed=1)
    c = kernel_shap(toy_f, 12, nsamples=200, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def additive_f(masks):
    return 0.1 * np.asarray(masks)[:, :15].sum(axis=1)


def test_all_30_regions_are_kept():
    values = kernel_shap(additive_f, 30, nsamples=300, seed=0)
    assert np.allclose(values[:15], 0.1, atol=1e-6)
    assert np.allclose(values[15:], 0.0, atol=1e-6)


def test_the_library_default_would_zero_out_regions():
    explainer = shap.KernelExplainer(additive_f, np.zeros((1, 30)))
    np.random.seed(0)
    default = np.asarray(explainer.shap_values(np.ones((1, 30)), nsamples=300, silent=True))
    assert np.count_nonzero(np.abs(default.reshape(-1)) > 1e-9) <= 10
