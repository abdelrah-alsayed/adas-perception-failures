import numpy as np
from PIL import Image

from explain_adas.explain_case import make_model_fn
from explain_adas.kernel_shap import kernel_shap

RED = (200, 0, 0)


def red_picture():
    return Image.new("RGB", (100, 30), RED)


def red_share(pictures):
    """A fake model: the score is the share of pixels that are still red."""
    return [float((np.array(p)[:, :, 0] == 200).mean()) for p in pictures]


def test_all_visible_scores_one_and_all_hidden_scores_zero():
    fn = make_model_fn(red_share, red_picture(), "grey")
    assert fn(np.ones((1, 30)))[0] == 1.0
    assert fn(np.zeros((1, 30)))[0] == 0.0


def test_one_row_per_coalition():
    fn = make_model_fn(red_share, red_picture(), "grey")
    masks = np.array([[1.0] * 30, [0.0] * 30, [1.0] * 15 + [0.0] * 15])
    assert np.allclose(fn(masks), [1.0, 0.0, 0.5])


def test_kernel_shap_through_real_masking_gives_each_region_its_area_share():
    fn = make_model_fn(red_share, red_picture(), "grey")
    values = kernel_shap(fn, 30, nsamples=300, seed=0)
    assert np.allclose(values, 1 / 30, atol=1e-6)
