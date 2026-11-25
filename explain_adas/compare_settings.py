import csv
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

from explain_adas.compare import rank_correlation, top_k_overlap
from explain_adas.kitti_labels import read_labels
from explain_adas.regions import region_boxes

IMG = "data/kitti/data_object_image_2/training/image_2/{}.png"
LAB = "data/kitti/data_object_label_2/training/label_2/{}.txt"
BASE = ("blur", 1, 2000)  # masking, KernelSHAP seed, budget
COMPARISONS = {
    "ks_seed_2": ("blur", 2, 2000),
    "budget_1000": ("blur", 1, 1000),
    "grey_mask": ("grey", 1, 2000),
}
# The sampling noise at budget 1,000: seed 1 against seed 2, both at budget 1,000 (added 2026-11-25).
# A 1,000 map is noisier than a 2,000 map, so the budget change must be read against this floor.
NOISE_FLOOR_BUDGET_1000 = ("ks_seed_2_budget_1000", ("blur", 1, 1000), ("blur", 2, 1000))
KS = (1, 3, 5)  # fixed before looking at results (decided 2026-11-22)
N_REGIONS = 30
FIELDS = ["pair", "role", "comparison", "spearman", "top1_overlap", "top3_overlap", "top5_overlap",
          "top3_ctx_overlap", "top5_ctx_overlap"]


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


def object_region(box, boxes):
    """Index of the region that holds the centre of the object box."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for i, (left, top, right, bottom) in enumerate(boxes):
        if left <= cx < right and top <= cy < bottom:
            return i


def case_skip(case):
    """Region number of the object of one row of splits/cases.csv."""
    box = read_labels(LAB.format(case["id"]))[int(case["line"])]["box"]
    width, height = Image.open(IMG.format(case["id"])).size
    return object_region(box, region_boxes(width, height))


def top_k_overlap_without(a, b, skip, k):
    """Like top_k_overlap, but region `skip` is removed from both lists."""
    a, b = list(a), list(b)
    a[skip] = b[skip] = float("-inf")  # it can never be in the top k
    return top_k_overlap(a, b, k)


def measure_row(name, a, b, skip):
    """All the measures for one comparison of two maps."""
    row = {"comparison": name, "spearman": rank_correlation(a, b)}
    for k in KS:
        row[f"top{k}_overlap"] = top_k_overlap(a, b, k=k)
    for k in (3, 5):
        row[f"top{k}_ctx_overlap"] = top_k_overlap_without(a, b, skip, k)
    return row


def compare_case(ks_case, baseline_case, skip):
    """Compare the base map of one case with the other settings and with hide-one (blur).

    The last row is the sampling noise at budget 1,000 (two maps that both differ from the base).
    """
    base = ks_case[BASE]
    others = {name: ks_case[setting] for name, setting in COMPARISONS.items()}
    others["hide_one_blur"] = baseline_case["blur"]
    result = [measure_row(name, base, other, skip) for name, other in others.items()]
    floor_name, setting_a, setting_b = NOISE_FLOOR_BUDGET_1000
    result.append(measure_row(floor_name, ks_case[setting_a], ks_case[setting_b], skip))
    return result


def main(model="seed_0"):
    ks = load_ks_maps(f"results/kernelshap_{model}.csv")
    baseline = load_baseline_maps(f"results/baseline_{model}.csv")
    with open("splits/cases.csv", newline="") as f:
        cases = {(c["pair"], c["role"]): c for c in csv.DictReader(f)}
    rows = []
    for (pair, role) in sorted(ks, key=lambda k: (int(k[0]), k[1])):
        case = cases[(pair, role)]
        skip = case_skip(case)
        for row in compare_case(ks[(pair, role)], baseline[(pair, role)], skip):
            rows.append({"pair": pair, "role": role, **row})
    out = f"results/settings_check_{model}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("saved", out, len(rows), "rows")

    print("chance level (two random lists share on average):",
          ", ".join(f"top{k} {k * k / N_REGIONS:.2f}" for k in KS),
          "| without object region:", ", ".join(f"top{k} {k * k / (N_REGIONS - 1):.2f}" for k in (3, 5)))
    print(f"{'comparison':<23}{'role':<6}{'median rho':>11}{'top1 same':>10}{'top3 med':>10}{'top5 med':>10}{'ctx3 med':>10}{'ctx5 med':>10}{'nan':>5}")
    for name in [*COMPARISONS, "hide_one_blur", NOISE_FLOOR_BUDGET_1000[0]]:
        for role in ("miss", "hit"):
            sel = [r for r in rows if r["comparison"] == name and r["role"] == role]
            rho = np.array([r["spearman"] for r in sel])
            top1 = np.mean([r["top1_overlap"] for r in sel])
            top3 = np.median([r["top3_overlap"] for r in sel])
            top5 = np.median([r["top5_overlap"] for r in sel])
            ctx3 = np.median([r["top3_ctx_overlap"] for r in sel])
            ctx5 = np.median([r["top5_ctx_overlap"] for r in sel])
            print(f"{name:<23}{role:<6}{np.nanmedian(rho):>11.2f}{top1:>10.0%}{top3:>10.1f}{top5:>10.1f}{ctx3:>10.1f}{ctx5:>10.1f}"
                  f"{int(np.isnan(rho).sum()):>5}")


if __name__ == "__main__":
    main(*sys.argv[1:])
