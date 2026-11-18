import csv
import io

from PIL import Image

from explain_adas.build_dataset import build_split


def test_build_split_writes_label_picture_and_csv_row(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    out_dir = tmp_path / "out"
    image_dir.mkdir()
    label_dir.mkdir()
    Image.new("RGB", (200, 100)).save(image_dir / "000001.png")
    (label_dir / "000001.txt").write_text(
        "Car 0.00 0 1.85 0 0 100 50 1 1 1 0 0 0 0\n"
        "DontCare -1 -1 -10 10 10 20 20 -1 -1 -1 -1000 -1000 -1000 -10\n"
    )
    buffer = io.StringIO()
    build_split("val", ["000001"], str(image_dir), str(label_dir),
                str(out_dir), csv.writer(buffer))
    label = (out_dir / "labels" / "val" / "000001.txt").read_text()
    assert label == "0 0.250000 0.250000 0.500000 0.500000\n"
    assert (out_dir / "images" / "val" / "000001.png").exists()
    assert buffer.getvalue().strip().split(",")[:5] == ["000001", "val", "0", "Car", "0"]