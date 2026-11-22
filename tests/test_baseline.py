from PIL import Image

from explain_adas.baseline import hide_one_drops

RED = (200, 0, 0)


def picture():
    return Image.new("RGB", (100, 30), RED)


def fake_f(image):
    """A fake model: the score is high only while the top left pixel is still red."""
    return 0.9 if image.getpixel((5, 5)) == RED else 0.1


def test_one_drop_per_region_and_the_full_score_is_returned():
    full, drops = hide_one_drops(fake_f, picture(), "grey")
    assert full == 0.9
    assert len(drops) == 30


def test_only_the_region_that_matters_has_a_drop():
    _, drops = hide_one_drops(fake_f, picture(), "grey")
    assert abs(drops[0] - 0.8) < 1e-9
    assert all(d == 0 for d in drops[1:])


def test_a_score_that_rises_gives_a_negative_drop():
    def odd_f(image):
        return 0.5 if image.getpixel((5, 5)) == RED else 0.7
    _, drops = hide_one_drops(odd_f, picture(), "grey")
    assert drops[0] < 0
