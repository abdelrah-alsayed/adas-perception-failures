"""Figures for the report.

Plain charts (no KITTI pictures) go to results/figures/ and are committed.
Maps with KITTI pictures go to results/maps/ and are NOT committed (license).
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from explain_adas.compare_models import MODELS, load_f_full  # noqa: E402
from explain_adas.compare_settings import BASE, load_ks_maps  # noqa: E402
from explain_adas.kitti_labels import read_labels  # noqa: E402
from explain_adas.regions import region_boxes  # noqa: E402
from explain_adas.report_tables import (REMOVAL_CSVS, curves_from_csv, read_csv, MODEL_CSV,  # noqa: E402
                                        SETTINGS_CSV)

IMG = "data/kitti/data_object_image_2/training/image_2/{}.png"
LAB = "data/kitti/data_object_label_2/training/label_2/{}.txt"
FIGURES_DIR = "results/figures"
MAPS_DIR = "results/maps"
SEED_2_SETTING = ("blur", 2, 2000)
UNSTABLE = [("3", "miss"), ("4", "hit"), ("5", "hit")]       # least stable in the settings check
FLIPPED_EXAMPLES = [("5", "miss"), ("4", "miss")]            # miss for seed 0, hit for another copy


def median_curves(curves, role):
    """Median over cases of: top_down, bottom_up, and the mean of the random curves."""
    keys = [k for k in curves if k[1] == role]
    result = {"top_down": [], "bottom_up": [], "random (mean of 20)": []}
    for key in keys:
        result["top_down"].append(curves[key]["top_down"])
        result["bottom_up"].append(curves[key]["bottom_up"])
        random_curves = [v for name, v in curves[key].items() if name.startswith("random_")]
        result["random (mean of 20)"].append(np.mean(random_curves, axis=0))
    return {name: np.median(np.array(values), axis=0) for name, values in result.items()}


def plot_removal(path):
    """Median score f while regions are hidden one by one."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    colors = {"top_down": "tab:red", "bottom_up": "tab:blue", "random (mean of 20)": "tab:grey"}
    for row, (variant, csv_path) in enumerate(REMOVAL_CSVS.items()):
        curves = curves_from_csv(csv_path)
        for col, role in enumerate(("miss", "hit")):
            ax = axes[row][col]
            for name, values in median_curves(curves, role).items():
                ax.plot(range(len(values)), values, label=name, color=colors[name], lw=2)
            ax.set_title(f"{role}es, hide {variant}" if role == "miss" else f"{role}s, hide {variant}")
            ax.set_xlabel("number of regions hidden")
            ax.set_ylabel("median score f")
            ax.grid(alpha=0.3)
    axes[0][0].legend()
    fig.suptitle("Removal check (seed_0 model, blur): median over 10 cases per panel")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_stability(path):
    """Median top-k overlap: noise floor against other settings and against other models."""
    settings_rows, model_rows = read_csv(SETTINGS_CSV), read_csv(MODEL_CSV)
    groups = [("KS seed\n(noise floor)", [r for r in settings_rows if r["comparison"] == "ks_seed_2"]),
              ("budget\n1000", [r for r in settings_rows if r["comparison"] == "budget_1000"]),
              ("grey\nmasking", [r for r in settings_rows if r["comparison"] == "grey_mask"]),
              ("hide-one", [r for r in settings_rows if r["comparison"] == "hide_one_blur"]),
              ("other\nmodels", model_rows)]
    panels = [("top3_overlap", 3, 3 * 3 / 30, "top-3 overlap (of 3)"),
              ("top5_overlap", 5, 5 * 5 / 30, "top-5 overlap (of 5)"),
              ("top3_ctx_overlap", 3, 3 * 3 / 29, "top-3 without the object cell"),
              ("top5_ctx_overlap", 5, 5 * 5 / 29, "top-5 without the object cell")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (field, top, chance, title) in zip(axes.flat, panels):
        medians = [np.median([float(r[field]) for r in rows]) for _, rows in groups]
        ax.bar([name for name, _ in groups], medians, color=["tab:grey"] + ["tab:blue"] * 3 + ["tab:red"])
        ax.axhline(chance, color="black", ls="--", lw=1)
        ax.text(4.45, chance, " chance", va="bottom", ha="right", fontsize=8)
        ax.set_ylim(0, top)
        ax.set_title(title)
        ax.set_ylabel("median overlap")
    fig.suptitle("Stability: how much of the top regions survives a change (all 20 cases)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def draw_map(ax, image, box, values, title, scale):
    """Picture + object box + the 10 x 3 grid coloured by value (red = helps, blue = lowers)."""
    ax.imshow(image)
    ranks = {int(r): i + 1 for i, r in enumerate(np.argsort(values)[::-1])}
    for i, (left, top, right, bottom) in enumerate(region_boxes(*image.size)):
        x = values[i] / max(scale, 1e-9)
        color = (1, 0, 0, min(abs(x), 1) * 0.65) if x > 0 else (0, 0.3, 1, min(abs(x), 1) * 0.65)
        ax.add_patch(Rectangle((left, top), right - left, bottom - top, facecolor=color,
                               edgecolor="white", lw=0.5))
        label = f"{values[i]:+.3f}" + (f"\n#{ranks[i]}" if ranks[i] <= 5 else "")
        ax.text((left + right) / 2, (top + bottom) / 2, label, color="white", ha="center", va="center", fontsize=6.5)
    ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor="lime", lw=2))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def load_case(case):
    image = Image.open(IMG.format(case["id"])).convert("RGB")
    box = read_labels(LAB.format(case["id"]))[int(case["line"])]["box"]
    return image, box


def plot_map_grid(panels, ncols, path):
    """panels: list of (case, values, title, scale)."""
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 3.6 * nrows), squeeze=False)
    for ax, (case, values, title, scale) in zip(axes.flat, panels):
        image, box = load_case(case)
        draw_map(ax, image, box, values, title, scale)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def pictures_available(cases):
    return all(os.path.exists(IMG.format(c["id"])) for c in cases)


def make_map_figures(cases):
    """The maps that need the KITTI pictures. Returns the list of files written."""
    by_key = {(c["pair"], c["role"]): c for c in cases}
    seed0 = load_ks_maps("results/kernelshap_seed_0.csv")
    os.makedirs(MAPS_DIR, exist_ok=True)
    written = []

    panels = []                                   # 1. the least stable cases: seed 1 next to seed 2
    for key in UNSTABLE:
        a, b = np.array(seed0[key][BASE]), np.array(seed0[key][SEED_2_SETTING])
        scale = max(abs(a).max(), abs(b).max())
        for seed, values in ((1, a), (2, b)):
            title = f"pair {key[0]} {key[1]}: KernelSHAP seed {seed}"
            panels.append((by_key[key], values, title, scale))
    path = os.path.join(MAPS_DIR, "maps_unstable_cases_seed_1_vs_2.png")
    plot_map_grid(panels, 2, path)
    written.append(path)

    model_maps = {name: load_ks_maps(p) for name, p in MODELS.items()}      # 2. same object, 3 models
    f_full = {name: load_f_full(p) for name, p in MODELS.items()}
    panels = []
    for key in FLIPPED_EXAMPLES:
        values = {name: np.array(model_maps[name][key][BASE]) for name in MODELS}
        scale = max(abs(v).max() for v in values.values())
        for name in MODELS:
            f = f_full[name][key]
            title = f"pair {key[0]} {key[1]}, model {name}: f = {f:.3f} ({'hit' if f >= 0.25 else 'miss'})"
            panels.append((by_key[key], values[name], title, scale))
    path = os.path.join(MAPS_DIR, "maps_three_models_flipped_cases.png")
    plot_map_grid(panels, 3, path)
    written.append(path)

    for case in cases:                            # 3. one map per case (seed_0 model, base setting)
        key = (case["pair"], case["role"])
        values = np.array(seed0[key][BASE])
        title = (f"pair {case['pair']} {case['role']} ({case['type']}, f = {float(case['target_score']):.3f}): "
                 f"KernelSHAP, blur, seed 1, budget 2000")
        path = os.path.join(MAPS_DIR, f"case_{int(case['pair']):02d}_{case['role']}.png")
        plot_map_grid([(case, values, title, np.abs(values).max())], 1, path)
        written.append(path)
    return written


def read_cases(path="splits/cases.csv"):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def make_figures():
    """Plain charts always; picture maps only if the KITTI pictures are on disk."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    written = []
    for name, function in (("removal_curves.png", plot_removal), ("stability_bars.png", plot_stability)):
        path = os.path.join(FIGURES_DIR, name)
        function(path)
        written.append(path)
    cases = read_cases()
    if pictures_available(cases):
        written += make_map_figures(cases)
    else:
        print("KITTI pictures not found: the map figures were skipped (see README).")
    return written
