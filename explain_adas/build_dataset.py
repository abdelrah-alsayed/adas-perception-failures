import os 
import shutil

from PIL import Image

from explain_adas.kitti_labels import read_labels
from explain_adas.kitti_to_yolo import CLASS_IDS, to_yolo_line


def link_or_copy(src, dst):
    """Hard-link the file (no extra disk space); copy it if linking fails."""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_split(split, ids, image_dir, label_dir, out_dir, writer):
    """Write YOLO images and labels for one split, and one csv row per object."""
    img_out = f"{out_dir}/images/{split}"
    lab_out = f"{out_dir}/labels/{split}"
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lab_out, exist_ok=True)
    for pid in ids:
        src = f"{image_dir}/{pid}.png"
        with Image.open(src) as im:
            width, height = im.size
        link_or_copy(src, f"{img_out}/{pid}.png")
        lines = []
        for k, obj in enumerate(read_labels(f"{label_dir}/{pid}.txt")):
            if obj["type"] not in CLASS_IDS:
                continue
            class_id = CLASS_IDS[obj["type"]]
            lines.append(to_yolo_line(class_id, obj["box"], width, height))
            writer.writerow([pid, split, k, obj["type"], class_id, obj["truncated"],
                             obj["occluded"], *obj["box"], width, height])
        with open(f"{lab_out}/{pid}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")