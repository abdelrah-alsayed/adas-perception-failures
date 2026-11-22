import numpy as np


def removal_masks(order, n_regions=30):
    """A table with len(order) + 1 rows for the model function (1 = visible, 0 = hidden).

    Row k hides the first k regions of `order`. Row 0 hides nothing. A region that
    is not in `order` stays visible in every row.
    """
    masks = np.ones((len(order) + 1, n_regions))
    for k in range(1, len(order) + 1):
        masks[k:, order[k - 1]] = 0   # this region stays hidden from row k on
    return masks


def make_orders(values, n_random, seed, skip=None):
    """The orders in which regions are hidden: top_down, bottom_up and n_random random ones.

    If `skip` is a region number, that region is left out of every order.
    """
    values = np.asarray(values)
    regions = np.array([r for r in range(len(values)) if r != skip])
    by_value = np.argsort(values[regions], kind="stable")   # smallest value first
    orders = {"top_down": [int(r) for r in regions[by_value][::-1]],
              "bottom_up": [int(r) for r in regions[by_value]]}
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        orders[f"random_{i}"] = [int(r) for r in regions[rng.permutation(len(regions))]]
    return orders


def curve_area(f_values, f_full):
    """Mean of f_k / f_full over all k. Lower = the score falls faster."""
    return float(np.mean(np.asarray(f_values, dtype=float) / f_full))


def case_summary(curves, f_full):
    """Compare the two ordered curves with the random curves of one case.

    curves: {order name: [f_0, f_1, ..., f_30]}
    """
    area = {name: curve_area(f, f_full) for name, f in curves.items()}
    random_areas = np.array([a for name, a in area.items() if name.startswith("random_")])
    return {"auc_top_down": area["top_down"],
            "auc_bottom_up": area["bottom_up"],
            "auc_random_mean": float(random_areas.mean()),
            "n_random": len(random_areas),
            # random orders whose score falls SLOWER than top_down (more is better for top_down)
            "top_down_beats": int((random_areas > area["top_down"]).sum()),
            # random orders whose score falls FASTER than bottom_up (more = bottom_up keeps f higher)
            "bottom_up_beats": int((random_areas < area["bottom_up"]).sum()),
            "bottom_up_peak": float(max(curves["bottom_up"]) / f_full)}


def case_summary_absolute(curves):
    """Like case_summary, but with the plain scores f (no division by f_full).

    curves: {order name: [f_0, f_1, ..., f_last]}
    """
    mean = {name: float(np.mean(f)) for name, f in curves.items()}
    random_means = np.array([m for name, m in mean.items() if name.startswith("random_")])
    return {"mean_top_down": mean["top_down"],
            "mean_bottom_up": mean["bottom_up"],
            "mean_random": float(random_means.mean()),
            "n_random": len(random_means),
            # random orders whose mean score is HIGHER than top_down (more is better for top_down)
            "top_down_beats": int((random_means > mean["top_down"]).sum()),
            # random orders whose mean score is LOWER than bottom_up (more = bottom_up keeps f higher)
            "bottom_up_beats": int((random_means < mean["bottom_up"]).sum()),
            "f_end": float(curves["top_down"][-1]),
            "bottom_up_peak": float(max(curves["bottom_up"]))}
