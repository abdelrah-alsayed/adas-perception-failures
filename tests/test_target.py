from explain_adas.target import target_score

LABEL = [100, 100, 140, 200]


def det(name, score, box=(100, 100, 140, 200)):
    return {"name": name, "score": score, "box": list(box)}


def test_takes_the_best_same_class_box_that_overlaps():
    detections = [det("Pedestrian", 0.3), det("Pedestrian", 0.7)]
    assert target_score(detections, LABEL, "Pedestrian") == 0.7


def test_other_classes_are_ignored():
    assert target_score([det("Cyclist", 0.9)], LABEL, "Pedestrian") == 0.0


def test_boxes_that_do_not_overlap_enough_are_ignored():
    far = det("Pedestrian", 0.9, (300, 100, 340, 200))
    assert target_score([far], LABEL, "Pedestrian") == 0.0


def test_no_detections_gives_zero():
    assert target_score([], LABEL, "Pedestrian") == 0.0


def test_the_limit_can_be_changed():
    half = det("Pedestrian", 0.5, (100, 100, 140, 150))
    assert target_score([half], LABEL, "Pedestrian") == 0.5
    assert target_score([half], LABEL, "Pedestrian", iou_limit=0.6) == 0.0
