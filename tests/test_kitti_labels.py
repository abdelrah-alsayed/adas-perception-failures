from explain_adas.kitti_labels import read_labels

def test_read_labels_reads_one_pedestrian(tmp_path):
    label_file = tmp_path / "000000.txt"
    label_file.write_text(
        "Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 "
        "1.89 0.48 1.20 1.84 1.47 8.41 0.01\n"
    )
    objects = read_labels(label_file)
    assert len(objects) == 1
    assert objects[0]["type"] == "Pedestrian"
    assert objects[0]["box"] == [712.40, 143.00, 810.73, 307.92]