from explain_adas.regions import apply_mask, make_filler, region_boxes


def hide_one_drops(f, image, method):
    """Hide each region alone. Return f(full picture) and the score drop for every region."""
    boxes = region_boxes(*image.size)
    filler = make_filler(image, method)
    full = f(image)
    drops = []
    for k in range(len(boxes)):
        keep = [True] * len(boxes)
        keep[k] = False
        drops.append(full - f(apply_mask(image, keep, boxes, filler)))
    return full, drops
