from PIL import Image, ImageFilter

COLS, ROWS = 10, 3
GREY = (114, 114, 114)


def region_boxes(width, height, cols=COLS, rows=ROWS):
    """Cut a picture into cols x rows boxes (left, top, right, bottom), row by row."""
    xs = [round(i * width / cols) for i in range(cols + 1)]
    ys = [round(j * height / rows) for j in range(rows + 1)]
    return [(xs[i], ys[j], xs[i + 1], ys[j + 1])
            for j in range(rows) for i in range(cols)]


def make_filler(image, method, rows=ROWS):
    """The picture whose pixels replace the hidden regions."""
    if method == "grey":
        return Image.new("RGB", image.size, GREY)
    if method == "blur":
        sigma = image.height / rows / 2
        return image.filter(ImageFilter.GaussianBlur(sigma))
    raise ValueError(f"unknown masking method: {method}")


def apply_mask(image, keep, boxes, filler):
    """Copy of the picture; regions with keep[k] == False are taken from the filler."""
    result = image.copy()
    for k, box in enumerate(boxes):
        if not keep[k]:
            result.paste(filler.crop(box), box[:2])
    return result
