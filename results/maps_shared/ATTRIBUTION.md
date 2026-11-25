# Attribution for the KITTI-based figures in this folder

The figures in this folder show pictures from the KITTI Vision Benchmark Suite (2D object detection, left colour images).

- **Source:** A. Geiger, P. Lenz and R. Urtasun, "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite", CVPR 2012. https://www.cvlibs.net/datasets/kitti/
- **License of the KITTI data:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0). https://creativecommons.org/licenses/by-nc-sa/3.0/
- **Changes made by this project:** the pictures are scaled and shown with a coloured 10 x 3 grid, numbers and a green box drawn over them.
- **License of these figure files:** the same, CC BY-NC-SA 3.0. Non-commercial use only. The code in the rest of this repository has its own license.
- **Purpose:** non-commercial research and study.

## Pictures shown (file names in `training/image_2/` of the KITTI object data)

| KITTI picture | Used in | What it shows |
|---|---|---|
| 006395.png | `case_05_miss.png`, `maps_three_models_flipped_cases.png` | pair 5, missed pedestrian |
| 007199.png | `case_05_hit.png` | pair 5, matched detected pedestrian |
| 000964.png | `maps_three_models_flipped_cases.png` | pair 4, missed pedestrian |
| 000073.png | `case_06_miss.png`, `maps_three_models_flipped_cases.png` | pair 6, missed cyclist |
| 001145.png | `case_06_hit.png` | pair 6, matched detected cyclist |

The other maps that `python -m explain_adas.make_report` writes to `results/maps/` are not shared (that folder is ignored by Git).
