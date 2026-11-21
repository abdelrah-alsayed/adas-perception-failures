import pytest
from PIL import Image

from explain_adas.regions import apply_mask, make_filler, region_boxes

RED = (200, 0, 0)


def red_picture():
    return Image.new("RGB", (100, 30), RED)


def test_30_boxes_tile_the_picture_exactly():
    boxes = region_boxes(1224, 370)
    assert len(boxes) == 30
    assert sum((r - l) * (b - t) for l, t, r, b in boxes) == 1224 * 370
    assert boxes[0][:2] == (0, 0)
    assert boxes[-1][2:] == (1224, 370)


def test_boxes_go_row_by_row():
    boxes = region_boxes(1242, 375)
    assert boxes[1][0] == boxes[0][2]
    assert boxes[10][1] == boxes[0][3]


def test_keeping_everything_changes_nothing():
    image = red_picture()
    result = apply_mask(image, [True] * 30, region_boxes(100, 30), make_filler(image, "grey"))
    assert list(result.getdata()) == list(image.getdata())


def test_a_hidden_region_is_filled_and_the_others_stay():
    image = red_picture()
    keep = [True] * 30
    keep[0] = False
    result = apply_mask(image, keep, region_boxes(100, 30), make_filler(image, "grey"))
    assert result.getpixel((1, 1)) == (114, 114, 114)
    assert result.getpixel((99, 29)) == RED


def test_the_original_picture_is_not_changed():
    image = red_picture()
    apply_mask(image, [False] * 30, region_boxes(100, 30), make_filler(image, "grey"))
    assert image.getpixel((1, 1)) == RED


def test_blur_keeps_a_flat_picture_flat():
    assert make_filler(red_picture(), "blur").getpixel((50, 15)) == RED


def test_blur_spreads_a_bright_spot():
    image = Image.new("RGB", (100, 30), (0, 0, 0))
    image.putpixel((50, 15), (255, 255, 255))
    assert make_filler(image, "blur").getpixel((50, 15))[0] < 255


def test_unknown_method_is_an_error():
    with pytest.raises(ValueError):
        make_filler(red_picture(), "mystery")
