from pathlib import Path

DATA = "data/yolo_kitti/dataset.yaml"
WEIGHTS = "data/weights/yolo26n.pt"
RUNS_DIR = "data/runs"


def make_train_args(seed, epochs, name):
    """All training settings in one place. Nothing is left to 'auto'."""
    return {
        "data": DATA, "epochs": epochs, "imgsz": 640, "batch": 8,
        "device": 0, "workers": 2, "seed": seed, "deterministic": True,
        "amp": False, "cache": "disk", "optimizer": "AdamW", "lr0": 0.001429,
        "close_mosaic": 10, "patience": 1000,
        "project": str(Path(RUNS_DIR).resolve()), "name": name,
    }


def train(seed, epochs, name):
    """Fine-tune yolo26n on our KITTI dataset. Return the results folder."""
    from ultralytics import YOLO

    model = YOLO(WEIGHTS)
    model.train(**make_train_args(seed, epochs, name))
    return Path(RUNS_DIR).resolve() / name


if __name__ == "__main__":
    import sys

    train(seed=int(sys.argv[1]), epochs=int(sys.argv[2]), name=sys.argv[3])