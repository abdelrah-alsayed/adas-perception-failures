import numpy as np

from explain_adas.regions import apply_mask, make_filler, region_boxes


def make_model_fn(score_pictures, image, method, chunk=16):
    """Build the function KernelSHAP calls.

    It gets a table of 0/1 values (1 = region visible), hides the regions with 0,
    and returns one score per row. score_pictures turns a list of pictures into a
    list of scores (the real one runs the model; tests use a fake one).
    KernelSHAP sends up to thousands of rows at once, so we work in small chunks
    to keep memory low.
    """
    boxes = region_boxes(*image.size)
    filler = make_filler(image, method)

    def model_fn(masks):
        masks = np.asarray(masks)
        scores = []
        for start in range(0, len(masks), chunk):
            pictures = [apply_mask(image, [v > 0.5 for v in row], boxes, filler)
                        for row in masks[start:start + chunk]]
            scores.extend(score_pictures(pictures))
        return np.asarray(scores, dtype=float)

    return model_fn
