import csv
import sys
from collections import defaultdict

import numpy as np

from explain_adas.compare import rank_correlation, top_k_overlap

BASE = ("blur", 1, 2000)  # masking, KernelSHAP seed, budget
COMPARISONS = {
    "ks_seed_2": ("blur", 2, 2000),
    "budget_1000": ("blur", 1, 1000),
    "grey_mask": ("grey", 1, 2000),
}
FIELDS = ["pair", "role", "comparison", "spearman", "top3_overlap"]


def load_ks_maps(path):
    """Read the KernelSHAP CSV. Return {(pair, role): {(method, seed, budget): 30 values}}."""
    maps = defaultdict(lambda: defaultdict(lambda: [0.0] * 30))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            setting = (r["method"], int(r["ks_seed"]), int(r["budget"]))
            maps[(r["pair"], r["role"])][setting][int(r["region"])] = float(r["value"])
    return maps


def load_baseline_maps(path):
    """Read the hide-one CSV. Return {(pair, role): {method: 30 drops}}."""
    maps = defaultdict(lambda: defaultdict(lambda: [0.0] * 30))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            maps[(r["pair"], r["role"])][r["method"]][int(r["region"])] = float(r["drop"])
    return maps


def compare_case(ks_case, baseline_case):
    """Compare the base map of one case with the other settings and with hide-one (blur)."""
    base = ks_case[BASE]
    others = {name: ks_case[setting] for name, setting in COMPARISONS.items()}
    others["hide_one_blur"] = baseline_case["blur"]
    result = []
    for name, other in others.items():
        result.append({"comparison": name,
                       "spearman": rank_correlation(base, other),
                       "top3_overlap": top_k_overlap(base, other, k=3)})
    return result


def main(model="seed_0"):
    ks = load_ks_maps(f"data/results/kernelshap_{model}.csv")
    baseline = load_baseline_maps(f"data/results/baseline_{model}.csv")
    rows = []
    for (pair, role) in sorted(ks, key=lambda k: (int(k[0]), k[1])):
        for row in compare_case(ks[(pair, role)], baseline[(pair, role)]):
            rows.append({"pair": pair, "role": role, **row})
    out = f"data/results/settings_check_{model}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("saved", out, len(rows), "rows")

    print(f"{'comparison':<15}{'role':<6}{'median rho':>11}{'min rho':>9}{'median top3':>13}{'nan':>5}")
    for name in [*COMPARISONS, "hide_one_blur"]:
        for role in ("miss", "hit"):
            sel = [r for r in rows if r["comparison"] == name and r["role"] == role]
            rho = np.array([r["spearman"] for r in sel])
            top3 = [r["top3_overlap"] for r in sel]
            print(f"{name:<15}{role:<6}{np.nanmedian(rho):>11.2f}{np.nanmin(rho):>9.2f}"
                  f"{np.median(top3):>13.1f}{int(np.isnan(rho).sum()):>5}")


if __name__ == "__main__":
    main(*sys.argv[1:])
