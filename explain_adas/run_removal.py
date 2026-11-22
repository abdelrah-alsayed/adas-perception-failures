import csv
import os
import sys
import time

import numpy as np
from PIL import Image

from explain_adas.compare_models import load_f_full
from explain_adas.compare_settings import BASE, load_ks_maps
from explain_adas.detector import load_model
from explain_adas.explain_case import make_model_fn
from explain_adas.kitti_labels import read_labels
from explain_adas.removal import case_summary, make_orders, removal_masks
from explain_adas.run_kernel_shap import real_scorer

IMG = "data/kitti/data_object_image_2/training/image_2/{}.png"
LAB = "data/kitti/data_object_label_2/training/label_2/{}.txt"
N_RANDOM = 20
RANDOM_SEED = 42
METHOD = "blur"   # the masking used for the hiding (same as the base map)
FIELDS = ["pair", "role", "order", "k", "f"]


def main(model_name="seed_0"):
    maps = load_ks_maps(f"data/results/kernelshap_{model_name}.csv")
    f_full = load_f_full(f"data/results/kernelshap_{model_name}.csv")
    model = load_model(f"data/runs/{model_name}/weights/last.pt")
    with open("splits/cases.csv", newline="") as f:
        cases = list(csv.DictReader(f))
    os.makedirs("data/results", exist_ok=True)
    summaries = []
    with open(f"data/results/removal_{model_name}.csv", "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for case in cases:
            key = (case["pair"], case["role"])
            start = time.time()
            obj = read_labels(LAB.format(case["id"]))[int(case["line"])]
            image = Image.open(IMG.format(case["id"])).convert("RGB")
            fn = make_model_fn(real_scorer(model, obj["box"], obj["type"]), image, METHOD)
            orders = make_orders(maps[key][BASE], N_RANDOM, RANDOM_SEED)
            masks = np.vstack([removal_masks(order) for order in orders.values()])
            scores = fn(masks).reshape(len(orders), -1)   # one row of 31 scores per order
            curves = {name: scores[i].tolist() for i, name in enumerate(orders)}
            for name, curve in curves.items():
                for k, f in enumerate(curve):
                    writer.writerow({"pair": case["pair"], "role": case["role"],
                                     "order": name, "k": k, "f": f})
            out.flush()
            first = curves["top_down"][0]
            if abs(first - f_full[key]) > 1e-3:
                print("WARNING: f at k=0 differs from f_full", key, first, f_full[key], flush=True)
            summary = case_summary(curves, f_full[key])
            summaries.append((key, f_full[key], summary))
            print(f'{case["pair"]:>2} {case["role"]:<4} f={f_full[key]:.3f} '
                  f'auc top_down {summary["auc_top_down"]:.2f} random {summary["auc_random_mean"]:.2f} '
                  f'bottom_up {summary["auc_bottom_up"]:.2f} | top_down beats {summary["top_down_beats"]}/{N_RANDOM} '
                  f'bottom_up beats {summary["bottom_up_beats"]}/{N_RANDOM} '
                  f'peak {summary["bottom_up_peak"]:.2f} ({time.time() - start:.0f} s)', flush=True)
    print_medians(summaries)


def print_medians(summaries):
    print("\nmedians (lower AUC = score falls faster)")
    print(f"{'group':<8}{'top_down':>10}{'random':>8}{'bottom_up':>11}{'td beats':>10}{'bu beats':>10}{'bu peak':>9}")
    for role in ("miss", "hit"):
        rows = [s for (key, f, s) in summaries if key[1] == role]
        med = lambda name: np.median([r[name] for r in rows])
        print(f"{role:<8}{med('auc_top_down'):>10.2f}{med('auc_random_mean'):>8.2f}{med('auc_bottom_up'):>11.2f}"
              f"{med('top_down_beats'):>10.1f}{med('bottom_up_beats'):>10.1f}{med('bottom_up_peak'):>9.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
