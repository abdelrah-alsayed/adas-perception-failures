# Dataset version note

## Source

KITTI 2D object detection benchmark (Geiger, Lenz and Urtasun, CVPR 2012): https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d

I used the left colour images (7,481 PNG files) and the training labels (7,481 text files). The code expects them in `data/kitti/data_object_image_2/training/image_2/` and `data/kitti/data_object_label_2/training/label_2/`.

KITTI is under CC BY-NC-SA 3.0. This repository has no KITTI data except five maps in `results/maps_shared/`, which show KITTI pictures with attribution.

## Preparation

I dropped the `DontCare` areas (they stay unlabelled background) and kept the 4,016 pictures that have no van, truck, tram, misc or sitting-person label. The model has three classes: `Car` = 0, `Pedestrian` = 1, `Cyclist` = 2. The pictures are used as they are (lossless PNG, original size). Labels are converted to the YOLO text format with each picture's own size, and `truncated` and `occluded` stay in a side file, `data/yolo_kitti/objects.csv`.

## Splits

`splits/train_ids.txt` has 3,213 pictures and `splits/val_ids.txt` has 803. The split is random, 80 % and 20 %, seed 42, balanced by whether a picture has a pedestrian or a cyclist (1,472 pictures have one and 2,544 do not). `python -m explain_adas.make_splits` makes it. All three model copies use it. There is no test set, because the KITTI test labels are not public.

KITTI frames come from video sequences, so neighbouring frames may land in different splits. I did not check this.

| Split | Car | Pedestrian | Cyclist |
|---|---|---|---|
| train | 11,471 | 1,994 | 794 |
| val | 2,753 | 545 | 184 |

## The 20 cases

`splits/cases.csv` lists 10 missed and 10 matched detected objects, all from `val`, chosen with seed 42 and at most one per picture. The rules are in `report.md`. The list depends on the trained seed 0 model, so a retrained model would give a different list. Do not rerun `make_cases` if you want to keep these 20 objects.
