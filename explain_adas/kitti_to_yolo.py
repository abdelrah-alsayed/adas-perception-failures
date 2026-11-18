CLASS_IDS = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
DROPPED_TYPES = {"Van", "Truck", "Tram", "Misc", "Person_sitting"}

def is_clean(objects):
    """True if no object in the picture has a dropped type."""
    return all(o["type"] not in DROPPED_TYPES for o in objects)

def to_yolo_line(class_id, box, width, height):
    """Make one YOLO label line from a KITTI box [left, top, right, bottom]."""
    cx = (box[0] + box[2]) / 2 / width
    cy = (box[1] + box[3]) / 2 / height
    w = (box[2] - box[0]) / width
    h = (box[3] - box[1]) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"