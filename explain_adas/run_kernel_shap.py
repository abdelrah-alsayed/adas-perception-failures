import csv
import os
import sys
import time

import numpy as np
from PIL import Image

from explain_adas.detector import detect_many, load_model
from explain_adas.explain_case import make_model_fn
from explain_adas.kernel_shap import kernel_shap
from explain_adas.kitti_labels import read_labels
from explain_adas.regions import COLS
from explain_adas.target import target_score

IMG = "data/kitti/data_object_image_2/training/image_2/{}.png"
LAB = "data/kitti/data_object_label_2/training/label_2/{}.txt"
N_REGIONS = 30
SETTINGS = [(method, seed, budget)
            for method in ("blur", "grey") for seed in (1, 2) for budget in (1000, 2000)]
BASE = ("blur", 1, 2000)
FIELDS = ["pair", "role", "id", "line", "type", "method", "ks_seed", "budget",
          "f_full", "f_empty", "region", "row", "col", "value", "seconds"]


def real_scorer(model, box, name):
    """A function that turns pictures into target scores by running the model."""
    def score_pictures(pictures):
        detections = detect_many(model, pictures, conf=0.001, one_to_one=True)
        return [target_score(d, box, name) for d in detections]
    return score_pictures


def explain(model, image, box, name, method, ks_seed, budget):
    """KernelSHAP values for one case and one setting."""
    fn = make_model_fn(real_scorer(model, box, name), image, method)
    values = kernel_shap(fn, N_REGIONS, budget, ks_seed)
    f_full = float(fn(np.ones((1, N_REGIONS)))[0])
    f_empty = float(fn(np.zeros((1, N_REGIONS)))[0])
    return f_full, f_empty, values


def main(model_name, filters):
    only_base = "base" in filters
    wanted = [f for f in filters if ":" in f]
    settings = [BASE] if only_base else SETTINGS
    model = load_model(f"data/runs/{model_name}/weights/last.pt")
    with open("splits/cases.csv", newline="") as f:
        cases = [c for c in csv.DictReader(f) if not wanted or f"{c['pair']}:{c['role']}" in wanted]
    os.makedirs("data/results", exist_ok=True)
    suffix = "_partial" if filters else ""
    with open(f"data/results/kernelshap_{model_name}{suffix}.csv", "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for case in cases:
            obj = read_labels(LAB.format(case["id"]))[int(case["line"])]
            image = Image.open(IMG.format(case["id"])).convert("RGB")
            for method, ks_seed, budget in settings:
                start = time.time()
                f_full, f_empty, values = explain(model, image, obj["box"], obj["type"],
                                                  method, ks_seed, budget)
                seconds = round(time.time() - start, 1)
                for k, value in enumerate(values):
                    writer.writerow({"pair": case["pair"], "role": case["role"], "id": case["id"],
                                     "line": case["line"], "type": obj["type"], "method": method,
                                     "ks_seed": ks_seed, "budget": budget, "f_full": f_full,
                                     "f_empty": f_empty, "region": k, "row": k // COLS,
                                     "col": k % COLS, "value": float(value), "seconds": seconds})
                out.flush()
                print(case["pair"], case["role"], case["id"], method, ks_seed, budget, seconds, "s",
                      flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "seed_0", sys.argv[2:])
