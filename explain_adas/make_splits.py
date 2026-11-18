import os

from explain_adas.kitti_labels import read_labels
from explain_adas.kitti_to_yolo import is_clean
from explain_adas.splitting import split_ids

LABEL_DIR = "data/kitti/data_object_label_2/training/label_2"
SEED = 42
VAL_FRACTION = 0.2

def write_ids(path, ids):
    """Write one picture id per line."""
    with open(path, "w") as f:
        f.write("\n".join(ids) + "\n")


def main():
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(LABEL_DIR))
    clean_ids = []
    strata = {}
    for pid in ids:
        objects = read_labels(f"{LABEL_DIR}/{pid}.txt")
        if not is_clean(objects):
            continue
        clean_ids.append(pid)
        types = {o["type"] for o in objects}
        has_rare = "Pedestrian" in types or "Cyclist" in types 
        strata[pid] = "pedestrian_or_cyclist" if has_rare else "other"
    train, val = split_ids(clean_ids, strata, VAL_FRACTION, SEED)
    os.makedirs("splits", exist_ok=True)
    write_ids("splits/train_ids.txt", train)
    write_ids("splits/val_ids.txt", val)
    print(len(ids), "pictures,", len(clean_ids), "clean,", len(train), "train,", len(val), "val")


if __name__ == "__main__":
    main()
