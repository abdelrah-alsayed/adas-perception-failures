from explain_adas.cases import classify_object, is_eligible

OBJ = {"type": "Pedestrian", "box": [100, 100, 140, 200]}


def det(name, score, box=(100, 100, 140, 200)):
    return {"name": name, "score": score, "box": list(box)}


def make(type_, height=100, occluded=0, truncated=0.0):
    return {"type": type_, "box": [0, 0, 30, height],
            "occluded": occluded, "truncated": truncated}


def test_hit():
    assert classify_object(OBJ, [det("Pedestrian", 0.8)]) == "hit"


def test_low_score():
    assert classify_object(OBJ, [det("Pedestrian", 0.10)]) == "low_score"


def test_wrong_class():
    assert classify_object(OBJ, [det("Cyclist", 0.7)]) == "wrong_class"


def test_not_found_when_no_box_overlaps():
    far = det("Pedestrian", 0.9, (300, 100, 340, 200))
    assert classify_object(OBJ, [far]) == "not_found"


def test_not_found_when_only_a_weak_wrong_class_box():
    assert classify_object(OBJ, [det("Cyclist", 0.1)]) == "not_found"


def test_best_scoring_overlapping_box_decides():
    detections = [det("Pedestrian", 0.1), det("Cyclist", 0.6)]
    assert classify_object(OBJ, detections) == "wrong_class"


def test_no_detections_means_not_found():
    assert classify_object(OBJ, []) == "not_found"


def test_filter_accepts_a_normal_pedestrian():
    assert is_eligible(make("Pedestrian"))


def test_filter_rejects_cars():
    assert not is_eligible(make("Car"))


def test_filter_rejects_small_unknown_or_cut_off_objects():
    assert not is_eligible(make("Cyclist", height=39))
    assert not is_eligible(make("Cyclist", occluded=3))
    assert not is_eligible(make("Pedestrian", truncated=0.31))


def test_filter_accepts_the_borders():
    assert is_eligible(make("Pedestrian", height=40, occluded=2, truncated=0.30))
