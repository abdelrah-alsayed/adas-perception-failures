def read_labels(path):
    """Read one KITTI label file. Return a list of objects."""
    objects = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            objects.append({
                "type": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "box": [float(x) for x in parts[4:8]],
            })
    return objects