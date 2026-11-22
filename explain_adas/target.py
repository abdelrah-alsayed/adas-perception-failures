from explain_adas.cases import IOU_LIMIT
from explain_adas.detector import detect
from explain_adas.geometry import box_iou


def target_score(detections, label_box, class_name, iou_limit=IOU_LIMIT):
    """f: the best score of class_name among boxes that overlap the label box, else 0."""
    scores = [d["score"] for d in detections
              if d["name"] == class_name and box_iou(d["box"], label_box) >= iou_limit]
    return max(scores, default=0.0)


def score_picture(model, image, label_box, class_name):
    """Run the model (one-to-one head, very low confidence) and return f."""
    detections = detect(model, image, conf=0.001, one_to_one=True)
    return target_score(detections, label_box, class_name)
