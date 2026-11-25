# Dataset version note

## Source

- **KITTI 2D object detection benchmark** (Geiger, Lenz, Urtasun, CVPR 2012), from https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
- Downloaded by the author around 2026-11-15 to 2026-11-16 and unzipped to `data/kitti/` (not in this repository).
- Files used: the **left colour images** (`data_object_image_2`, folder `training/image_2/`, 7,481 PNG files) and the **training labels** (`data_object_label_2`, folder `training/label_2/`, 7,481 text files).
- Not used: the 7,518 test pictures (they have no labels), the calibration files and the devkit.
- Not recorded: the exact zip file names, sizes and checksums (the zip files were deleted after unzipping, and the author decided not to keep checksums). Check that your copy has 7,481 pictures and 7,481 label files named `000000` to `007480`.
- The Ultralytics `kitti.zip` copy of KITTI was only inspected. It was **not** used for training or for the split.
- The licence is CC BY-NC-SA 3.0, academic use only, no redistribution. No KITTI pictures or raw labels are in this repository.

## What we did with it

1. **Dropped** the `DontCare` areas (they stay unlabelled background).
2. **Kept only "clean" pictures:** a picture with a Van, Truck, Tram, Misc or Person_sitting object is dropped. 7,481 pictures become **4,016**.
3. **Three classes**, with the KITTI spelling: `Car` = 0, `Pedestrian` = 1, `Cyclist` = 2 (`data/yolo_kitti/dataset.yaml`).
4. **Pictures are used as they are** (lossless PNG, original size; sizes differ a little, for example 1224 x 370 and 1242 x 375). Labels are converted to the YOLO text format using each picture's own size. The columns `truncated` and `occluded` are kept in a side file (`data/yolo_kitti/objects.csv`).

## Splits

- `splits/train_ids.txt`: **3,213** pictures. `splits/val_ids.txt`: **803** pictures. Together they are the 4,016 clean pictures.
- Rule: random split, 80 % / 20 %, seed 42, stratified by "the picture has a Pedestrian or Cyclist" (1,472 such pictures and 2,544 without). Made by `python -m explain_adas.make_splits`.
- All three model copies use the same split. There is no separate test set (the KITTI test labels are not public). Models are trained on `train` and the 20 cases come from `val`.
- KITTI frames come from video sequences, so neighbouring pictures may land in different splits. We did not check this. It is a limit.

## Objects in the dataset (from `objects.csv`)

| Split | Car | Pedestrian | Cyclist |
|---|---|---|---|
| train | 11,471 | 1,994 | 794 |
| val | 2,753 | 545 | 184 |

Total: 17,741 objects.

## The 20 cases

`splits/cases.csv`: 10 missed objects (5 Pedestrian, 5 Cyclist) and 10 matched correct detections, all from `val` pictures, chosen with seed 42 and one object per picture. The miss rule and the filter are in the report and in `explain_adas/cases.py` and `explain_adas/select_cases.py`. The list depends on the trained `seed_0` model, so a retrained model would give a different list. Do not rerun `make_cases` if you want to keep these 20 objects.
