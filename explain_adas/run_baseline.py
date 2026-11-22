import csv
import os
import sys

from PIL import Image

from explain_adas.baseline import hide_one_drops
from explain_adas.detector import load_model
from explain_adas.kitti_labels import read_labels
from explain_adas.regions import COLS
from explain_adas.target import score_picture

IMG = "data/kitti/data_object_image_2/training/image_2/{}.png"
LAB = "data/kitti/data_object_label_2/training/label_2/{}.txt"
METHODS = ("blur", "grey")
FIELDS = ["pair", "role", "id", "line", "type", "method", "full", "region", "row", "col", "drop"]


def main(seed):
    model = load_model(f"data/runs/{seed}/weights/last.pt")
    rows = []
    with open("splits/cases.csv", newline="") as f:
        cases = list(csv.DictReader(f))
    for case in cases:
        obj = read_labels(LAB.format(case["id"]))[int(case["line"])]
        image = Image.open(IMG.format(case["id"])).convert("RGB")

        def score(picture):
            return score_picture(model, picture, obj["box"], obj["type"])

        for method in METHODS:
            full, drops = hide_one_drops(score, image, method)
            for k, drop in enumerate(drops):
                rows.append({"pair": case["pair"], "role": case["role"], "id": case["id"],
                             "line": case["line"], "type": obj["type"], "method": method,
                             "full": full, "region": k, "row": k // COLS, "col": k % COLS,
                             "drop": drop})
        print("done", case["pair"], case["role"], case["id"])
    os.makedirs("data/results", exist_ok=True)
    with open(f"data/results/baseline_{seed}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "seed_0")
