from explain_adas.geometry import box_iou

IOU_LIMIT = 0.5
SCORE_LIMIT = 0.25


def classify_object(obj, detections, iou_limit=IOU_LIMIT, score_limit=SCORE_LIMIT):
    """Say what happened to one real object: hit, low_score, wrong_class or not_found."""
    overlapping = [d for d in detections if box_iou(obj["box"], d["box"]) >= iou_limit]
    if not overlapping:
        return "not_found"
    best = max(overlapping, key=lambda d: d["score"])
    same_class = best["name"] == obj["type"]
    confident = best["score"] >= score_limit
    if same_class and confident:
        return "hit"
    if same_class:
        return "low_score"
    if confident:
        return "wrong_class"
    return "not_found"


def is_eligible(obj):
    """Filter C: can this real object become a case?"""
    height = obj["box"][3] - obj["box"][1]
    return (obj["type"] in ("Pedestrian", "Cyclist")
            and height >= 40
            and obj["occluded"] <= 2
            and obj["truncated"] <= 0.30)
