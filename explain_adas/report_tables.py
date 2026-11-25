"""Tables for the report. They read only the files in results/ (no GPU, no pictures)."""
import csv
import os

import numpy as np

from explain_adas.compare_models import MODELS, is_flipped, load_f_full
from explain_adas.removal import case_summary_absolute

TABLES_DIR = "results/tables"
SETTINGS_CSV = "results/settings_check_seed_0.csv"
MODEL_CSV = "results/model_check.csv"
REMOVAL_CSVS = {"all regions": "results/removal_seed_0.csv",
                "context only": "results/removal_context_seed_0.csv"}
COMPARISON_NAMES = {"ks_seed_2": "KernelSHAP seed 2 (noise floor)",
                    "budget_1000": "budget 1000",
                    "ks_seed_2_budget_1000": "seed 1 vs seed 2 at budget 1000 (noise floor at 1000)",
                    "grey_mask": "grey masking",
                    "hide_one_blur": "hide-one baseline"}
SUMMARY_HEADER = ["n", "rho", "top-1 same", "top-3", "top-5", "ctx top-3", "ctx top-5"]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_markdown(header, rows):
    """A markdown table (a list of header names and a list of rows)."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def summary_cells(rows):
    """Medians of one group of comparison rows. Top-1 is the share of cases with the same top region."""
    def med(field):
        return f"{np.median([float(r[field]) for r in rows]):.2f}"
    top1 = np.mean([float(r["top1_overlap"]) for r in rows])
    return [len(rows), med("spearman"), f"{top1:.0%}", med("top3_overlap"), med("top5_overlap"),
            med("top3_ctx_overlap"), med("top5_ctx_overlap")]


def settings_table(settings_rows):
    """Base map against other settings, per role."""
    rows = []
    for name, label in COMPARISON_NAMES.items():
        for role in ("miss", "hit"):
            group = [r for r in settings_rows if r["comparison"] == name and r["role"] == role]
            rows.append([label, role] + summary_cells(group))
    return ["comparison", "role"] + SUMMARY_HEADER, rows


def model_table(model_rows, settings_rows):
    """Retrained models against the KernelSHAP seed noise floor, for all cases and the flipped ones."""
    flipped = {(r["pair"], r["role"]) for r in model_rows if r["flipped"] == "True"}
    noise = [r for r in settings_rows if r["comparison"] == "ks_seed_2"]
    rows = [["all 20 cases", "noise floor: same model, other KernelSHAP seed"] + summary_cells(noise),
            ["all 20 cases", "different models (3 pairs x 20 cases)"] + summary_cells(model_rows)]
    for pair in sorted({r["models"] for r in model_rows}):
        rows.append(["all 20 cases", "  " + pair] + summary_cells([r for r in model_rows if r["models"] == pair]))
    rows.append(["4 flipped cases", "noise floor: same model, other KernelSHAP seed"] +
                summary_cells([r for r in noise if (r["pair"], r["role"]) in flipped]))
    rows.append(["4 flipped cases", "different models"] + summary_cells([r for r in model_rows if r["flipped"] == "True"]))
    return ["cases", "comparison"] + SUMMARY_HEADER, rows


def flip_table():
    """f_full of every case under the three models, and whether they disagree on hit/miss."""
    scores = {name: load_f_full(path) for name, path in MODELS.items()}
    keys = sorted(scores["seed_0"], key=lambda k: (int(k[0]), k[1]))
    rows = []
    for key in keys:
        values = [scores[name][key] for name in MODELS]
        rows.append([key[0], key[1]] + [f"{v:.3f}" for v in values] + ["yes" if is_flipped(values) else ""])
    return ["pair", "role"] + [f"f_full {name}" for name in MODELS] + ["models disagree"], rows


def curves_from_csv(path):
    """Read a removal CSV. Return {(pair, role): {order name: [f_0, f_1, ...]}}."""
    curves = {}
    for r in read_csv(path):
        curves.setdefault((r["pair"], r["role"]), {}).setdefault(r["order"], []).append(float(r["f"]))
    return curves


def removal_summaries(curves):
    """One summary dict per case (absolute scores), sorted by pair and role."""
    result = []
    for key in sorted(curves, key=lambda k: (int(k[0]), k[1])):
        summary = case_summary_absolute(curves[key])
        summary.update({"pair": key[0], "role": key[1], "f_full": curves[key]["top_down"][0]})
        result.append(summary)
    return result


REMOVAL_FIELDS = ["f_full", "f_end", "mean_top_down", "mean_random", "mean_bottom_up",
                  "top_down_beats", "bottom_up_beats", "bottom_up_peak"]


def show(field, value):
    """Counts (beats) without decimals, scores with 3 decimals."""
    return f"{value:g}" if field.endswith("beats") else f"{value:.3f}"


def removal_case_table(summaries):
    rows = [[s["pair"], s["role"]] + [show(f, s[f]) for f in REMOVAL_FIELDS] for s in summaries]
    return ["pair", "role"] + REMOVAL_FIELDS, rows


def removal_summary_table(summaries_by_variant):
    """Medians per variant and role."""
    rows = []
    for variant, summaries in summaries_by_variant.items():
        for role in ("miss", "hit"):
            group = [s for s in summaries if s["role"] == role]
            rows.append([variant, role, len(group)] +
                        [show(f, np.median([s[f] for s in group])) for f in REMOVAL_FIELDS])
    return ["variant", "role", "n"] + REMOVAL_FIELDS, rows


def write_table(name, header, rows):
    """Save a table as CSV in results/tables/ and return it as markdown."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, f"{name}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return to_markdown(header, rows)


def make_tables():
    """Write every table. Return {name: markdown}."""
    settings_rows, model_rows = read_csv(SETTINGS_CSV), read_csv(MODEL_CSV)
    tables = {"settings_summary": settings_table(settings_rows),
              "model_summary": model_table(model_rows, settings_rows),
              "f_full_by_model": flip_table()}
    by_variant = {variant: removal_summaries(curves_from_csv(path)) for variant, path in REMOVAL_CSVS.items()}
    tables["removal_summary"] = removal_summary_table(by_variant)
    tables["removal_cases_all_regions"] = removal_case_table(by_variant["all regions"])
    tables["removal_cases_context_only"] = removal_case_table(by_variant["context only"])
    markdown = {name: write_table(name, header, rows) for name, (header, rows) in tables.items()}
    with open(os.path.join(TABLES_DIR, "tables.md"), "w", newline="") as f:
        f.write("# Tables (made by `python -m explain_adas.make_report`)\n\n")
        for name, text in markdown.items():
            f.write(f"## {name}\n\n{text}\n\n")
    return markdown
