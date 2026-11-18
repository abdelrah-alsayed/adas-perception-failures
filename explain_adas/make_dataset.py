import csv
import os

from explain_adas.build_dataset import build_split
from explain_adas.kitti_to_yolo import CLASS_IDS

IMAGE_DIR = "data/kitti/data_object_image_2/training/image_2"
LABEL_DIR = "data/kitti/data_object_label_2/training/label_2"
OUT_DIR = "data/yolo_kitti"
HEADER = ["id", "split", "line", "type", "class_id", "truncated", "occluded",
          "left", "top", "right", "bottom", "width", "height"]


def read_ids(path):
    """Read one picture id per line."""
    with open(path) as f:
        return f.read().split()


def write_yaml():
    """Write the file that tells Ultralytics where the data is."""
    names = {class_id: name for name, class_id in CLASS_IDS.items()}
    with open(f"{OUT_DIR}/dataset.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(OUT_DIR)}\n")
        f.write("train: images/train\nval: images/val\nnames:\n")
        for class_id in sorted(names):
            f.write(f"  {class_id}: {names[class_id]}\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/objects.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for split in ("train", "val"):
            ids = read_ids(f"splits/{split}_ids.txt")
            build_split(split, ids, IMAGE_DIR, LABEL_DIR, OUT_DIR, writer)
            print(split, len(ids), "pictures")
    write_yaml()
    print("done:", OUT_DIR)


if __name__ == "__main__":
    main()          