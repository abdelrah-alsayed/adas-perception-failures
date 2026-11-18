import pytest

from explain_adas.kitti_to_yolo import is_clean, to_yolo_line


def test_picture_with_only_kept_types_is_clean():
    objects = [{"type": "Car"}, {"type": "Pedestrian"}, {"type": "DontCare"}]
    assert is_clean(objects)


def test_picture_with_a_van_is_not_clean():
    objects = [{"type": "Car"}, {"type": "Van"}]
    assert not is_clean(objects)


def test_yolo_line_of_the_real_pedestrian():
    line = to_yolo_line(1, [712.40, 143.00, 810.73, 307.92], 1224, 370)
    assert line == "1 0.622194 0.609351 0.080335 0.445730"


def test_yolo_line_of_a_simple_box():
    line = to_yolo_line(0, [0, 0, 100, 100], 200, 200)
    assert line == "0 0.250000 0.250000 0.500000 0.500000"


@pytest.mark.parametrize("name", ["Van", "Truck", "Tram", "Misc", "Person_sitting"])
def test_every_dropped_type_makes_a_picture_not_clean(name):
    assert not is_clean([{"type": name}])
