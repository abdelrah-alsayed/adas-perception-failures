import csv
import os

from explain_adas.case_table import build_table
from explain_adas.detector import load_model
from explain_adas.select_cases import pick_matched_hits, pick_misses

IMAGE_DIR = "data/kitti/data_object_image_2/training/image_2"
LABEL_DIR = "data/kitti/data_object_label_2/training/label_2"
WEIGHTS = "data/runs/seed_0/weights/last.pt"
TABLE_PATH = "results/seed_0_val_table.csv"
CASES_PATH = "splits/cases.csv"
CASE_FIELDS = ["pair", "role", "id", "line", "type", "kind", "height",
               "occluded", "truncated", "target_score", "match"]


def read_table(path):
    """Read the table back, with numbers as numbers."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["line"] = int(r["line"])
            r["height"] = float(r["height"])
            r["occluded"] = int(r["occluded"])
            r["truncated"] = float(r["truncated"])
            r["target_score"] = float(r["target_score"])
            r["eligible"] = r["eligible"] == "True"
            rows.append(r)
    return rows


def case_rows(misses, pairs):
    """Turn the picked misses and their hits into rows of the case list."""
    hit_of = {(p["miss"]["id"], p["miss"]["line"]): p for p in pairs}
    out = []
    for number, miss in enumerate(misses, 1):
        out.append(dict(miss, pair=number, role="miss", match=""))
        pair = hit_of.get((miss["id"], miss["line"]))
        if pair:
            out.append(dict(pair["hit"], pair=number, role="hit", match=pair["match"]))
    return out


def main():
    with open("splits/val_ids.txt") as f:
        ids = f.read().split()
    os.makedirs("results", exist_ok=True)
    build_table(load_model(WEIGHTS), ids, IMAGE_DIR, LABEL_DIR, TABLE_PATH)
    rows = [r for r in read_table(TABLE_PATH) if r["eligible"]]
    misses = pick_misses(rows)
    pairs = pick_matched_hits(misses, rows)
    with open(CASES_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CASE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(case_rows(misses, pairs))
    print(len(rows), "eligible objects;", len(misses), "misses;", len(pairs), "pairs")


if __name__ == "__main__":
    main()
