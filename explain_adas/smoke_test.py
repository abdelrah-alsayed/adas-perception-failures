"""A quick check that the whole pipeline works: python -m explain_adas.smoke_test

Part A runs the unit tests (no GPU, no data needed).
Part B runs KernelSHAP with a small budget on 2 real cases and checks the numbers.
Part B is skipped, with a message, if the checkpoint or the KITTI pictures are missing.
"""
import csv
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from explain_adas.detector import load_model
from explain_adas.kitti_labels import read_labels
from explain_adas.run_kernel_shap import IMG, LAB, N_REGIONS, explain

CHECKPOINT = "data/runs/seed_0/weights/last.pt"
SMOKE_CASES = [("5", "miss"), ("5", "hit")]   # (pair, role) from splits/cases.csv
BUDGET = 200                                  # the real runs use 1,000 and 2,000
KS_SEED = 1
TOLERANCE = 1e-3


def check_values(stored_score, f_full, f_empty, values, repeat_values):
    """The checks for one case. Return a list of (name, ok, detail)."""
    values = np.asarray(values)
    return [
        ("f_full equals the score stored in splits/cases.csv",
         bool(abs(f_full - stored_score) < TOLERANCE), f"{f_full:.4f} vs {stored_score:.4f}"),
        ("f_empty (everything hidden) is 0", bool(abs(f_empty) < TOLERANCE), f"{f_empty:.4f}"),
        ("30 finite values", len(values) == N_REGIONS and bool(np.isfinite(values).all()), ""),
        ("values add up to f_full - f_empty",
         bool(abs(values.sum() - (f_full - f_empty)) < TOLERANCE), f"{values.sum():.4f} vs {f_full - f_empty:.4f}"),
        ("same seed gives the same values (second run)",
         bool(np.allclose(values, repeat_values, atol=TOLERANCE)),
         f"max difference {np.abs(values - np.asarray(repeat_values)).max():.1e}"),
    ]


def run_unit_tests():
    start = time.time()
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    summary = lines[-1] if lines else "no output"
    print(f"{'PASS' if result.returncode == 0 else 'FAIL'}  unit tests: {summary} ({time.time() - start:.0f} s)")
    return result.returncode == 0


def read_smoke_cases():
    with open("splits/cases.csv", newline="") as f:
        rows = {(r["pair"], r["role"]): r for r in csv.DictReader(f)}
    return [rows[key] for key in SMOKE_CASES]


def missing_for_mini_run(cases):
    """Why part B cannot run, or None if it can."""
    if not os.path.exists(CHECKPOINT):
        return f"checkpoint not found: {CHECKPOINT}"
    for case in cases:
        if not os.path.exists(IMG.format(case["id"])):
            return f"KITTI picture not found: {IMG.format(case['id'])}"
    return None


def run_mini_run(cases):
    model = load_model(CHECKPOINT)
    all_ok = True
    for case in cases:
        start = time.time()
        obj = read_labels(LAB.format(case["id"]))[int(case["line"])]
        image = Image.open(IMG.format(case["id"])).convert("RGB")
        args = (model, image, obj["box"], obj["type"], "blur", KS_SEED, BUDGET)
        f_full, f_empty, values = explain(*args)
        _, _, repeat = explain(*args)
        print(f"case pair {case['pair']} {case['role']} (picture {case['id']}), budget {BUDGET}, "
              f"{time.time() - start:.0f} s")
        for name, ok, detail in check_values(float(case["target_score"]), f_full, f_empty, values, repeat):
            print(f"  {'PASS' if ok else 'FAIL'}  {name} {('(' + detail + ')') if detail else ''}")
            all_ok = all_ok and ok
    return all_ok


def main():
    start = time.time()
    ok = run_unit_tests()
    cases = read_smoke_cases()
    reason = missing_for_mini_run(cases)
    if reason:
        print(f"SKIP  mini-run on real cases: {reason}")
    else:
        ok = run_mini_run(cases) and ok
    print(f"{'SMOKE TEST PASSED' if ok else 'SMOKE TEST FAILED'} ({time.time() - start:.0f} s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
