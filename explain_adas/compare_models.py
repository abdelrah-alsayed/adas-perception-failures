import csv
from itertools import combinations

import numpy as np

from explain_adas.compare import rank_correlation, top_k_overlap
from explain_adas.compare_settings import BASE, N_REGIONS, case_skip, load_ks_maps, top_k_overlap_without

HIT_LIMIT = 0.25  # same limit as the miss rule
MODELS = {
    "seed_0": "results/kernelshap_seed_0.csv",
    "seed_1": "results/kernelshap_seed_1_base.csv",
    "seed_2": "results/kernelshap_seed_2_base.csv",
}
FIELDS = ["pair", "role", "models", "flipped", "spearman", "top1_overlap", "top3_overlap",
          "top5_overlap", "top3_ctx_overlap", "top5_ctx_overlap"]


def load_f_full(path):
    """Read f_full of the base setting. Return {(pair, role): f_full}."""
    result = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if (r["method"], int(r["ks_seed"]), int(r["budget"])) == BASE:
                result[(r["pair"], r["role"])] = float(r["f_full"])
    return result


def is_flipped(scores, limit=HIT_LIMIT):
    """True if some models call the object a hit and others a miss."""
    return len({score >= limit for score in scores}) > 1


def compare_models_case(maps, skip):
    """Compare the base maps of every pair of models for one case.

    maps: {model name: 30 values}
    """
    result = []
    for name_a, name_b in combinations(maps, 2):
        a, b = maps[name_a], maps[name_b]
        result.append({"models": f"{name_a} vs {name_b}",
                       "spearman": rank_correlation(a, b),
                       "top1_overlap": top_k_overlap(a, b, k=1),
                       "top3_overlap": top_k_overlap(a, b, k=3),
                       "top5_overlap": top_k_overlap(a, b, k=5),
                       "top3_ctx_overlap": top_k_overlap_without(a, b, skip, 3),
                       "top5_ctx_overlap": top_k_overlap_without(a, b, skip, 5)})
    return result


def summary_line(label, rows):
    """One printed line with the medians of the rows (top-1 is the share of cases)."""
    def med(field):
        return np.median([float(r[field]) for r in rows])
    top1 = np.mean([float(r["top1_overlap"]) for r in rows])
    return (f"{label:<34}{len(rows):>4}{med('spearman'):>8.2f}{top1:>8.0%}{med('top3_overlap'):>7.1f}"
            f"{med('top5_overlap'):>7.1f}{med('top3_ctx_overlap'):>7.1f}{med('top5_ctx_overlap'):>7.1f}")


def main():
    maps = {name: load_ks_maps(path) for name, path in MODELS.items()}
    scores = {name: load_f_full(path) for name, path in MODELS.items()}
    with open("splits/cases.csv", newline="") as f:
        cases = {(c["pair"], c["role"]): c for c in csv.DictReader(f)}
    keys = sorted(cases, key=lambda k: (int(k[0]), k[1]))
    flipped = {k for k in keys if is_flipped([scores[m][k] for m in MODELS])}

    rows = []
    for key in keys:
        per_model = {name: maps[name][key][BASE] for name in MODELS}
        for row in compare_models_case(per_model, case_skip(cases[key])):
            rows.append({"pair": key[0], "role": key[1], "flipped": key in flipped, **row})
    with open("results/model_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("saved results/model_check.csv", len(rows), "rows")
    print("cases where the models disagree on hit/miss:", len(flipped), "of", len(keys))

    # the noise floor: same model (seed_0), two KernelSHAP seeds (Step 7.2)
    with open("results/settings_check_seed_0.csv", newline="") as f:
        noise = [r for r in csv.DictReader(f) if r["comparison"] == "ks_seed_2"]
    noise_flipped = [r for r in noise if (r["pair"], r["role"]) in flipped]

    print("chance level: top1 0.03, top3 0.30, top5 0.83; without object cell: top3 0.31, top5 0.86")
    print(f"{'group':<34}{'n':>4}{'rho':>8}{'top1':>8}{'top3':>7}{'top5':>7}{'ctx3':>7}{'ctx5':>7}")
    print(summary_line("NOISE FLOOR: same model, KS seed", noise))
    print(summary_line("DIFFERENT MODELS, all pairs", rows))
    for pair in ("seed_0 vs seed_1", "seed_0 vs seed_2", "seed_1 vs seed_2"):
        print(summary_line("  " + pair, [r for r in rows if r["models"] == pair]))
    print("--- only the cases where the models disagree on hit/miss")
    print(summary_line("NOISE FLOOR: same model, KS seed", noise_flipped))
    print(summary_line("DIFFERENT MODELS, all pairs", [r for r in rows if r["flipped"]]))
    print("(n counts case x model pair; the noise floor has one row per case)")


if __name__ == "__main__":
    main()
