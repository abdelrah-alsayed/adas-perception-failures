import csv

from explain_adas.cases import IOU_LIMIT, classify_object, is_eligible
from explain_adas.detector import detect
from explain_adas.geometry import box_iou
from explain_adas.kitti_labels import read_labels
from explain_adas.kitti_to_yolo import CLASS_IDS

FIELDS = ["id", "line", "type", "height", "occluded", "truncated",
          "kind", "target_score", "eligible"]


def object_row(pid, line, obj, detections):
    """One table row for one real object: what the model did with it."""
    overlapping = [d for d in detections if box_iou(obj["box"], d["box"]) >= IOU_LIMIT]
    same_class = [d["score"] for d in overlapping if d["name"] == obj["type"]]
    return {
        "id": pid,
        "line": line,
        "type": obj["type"],
        "height": obj["box"][3] - obj["box"][1],
        "occluded": obj["occluded"],
        "truncated": obj["truncated"],
        "kind": classify_object(obj, detections),
        "target_score": max(same_class, default=0.0),
        "eligible": is_eligible(obj),
    }


def build_table(model, ids, image_dir, label_dir, out_path):
    """Run the model (one-to-one head, very low confidence) on every picture."""
    rows = []
    for pid in ids:
        detections = detect(model, f"{image_dir}/{pid}.png", conf=0.001, one_to_one=True)
        for line, obj in enumerate(read_labels(f"{label_dir}/{pid}.txt")):
            if obj["type"] in CLASS_IDS:
                rows.append(object_row(pid, line, obj, detections))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
