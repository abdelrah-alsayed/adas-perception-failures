import pytest

from explain_adas.geometry import box_iou


def test_identical_boxes_have_iou_one():
    box = [10.0, 20.0, 50.0, 80.0]
    assert box_iou(box, box) == 1.0


def test_boxes_that_do_not_touch_have_iou_zero():
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_half_shifted_boxes():
    # two 10 x 10 boxes, one shifted by 5: overlap 50, union 150
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_real_pedestrian_prediction_and_label():
    label = [712.40, 143.00, 810.73, 307.92]
    pred = [712.7, 153.3, 804.3, 306.7]
    assert box_iou(label, pred) == pytest.approx(0.866, abs=0.001)