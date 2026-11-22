import numpy as np

from explain_adas.report_figures import median_curves
from explain_adas.report_tables import (curves_from_csv, removal_summaries, show, summary_cells,
                                        to_markdown)


def test_to_markdown_makes_header_separator_and_rows():
    text = to_markdown(["a", "b"], [[1, 2], [3, 4]])
    assert text.splitlines() == ["| a | b |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |"]


def test_summary_cells_use_medians_and_share_of_same_top_region():
    rows = [{"spearman": "0.5", "top1_overlap": "1", "top3_overlap": "2", "top5_overlap": "3",
             "top3_ctx_overlap": "1", "top5_ctx_overlap": "2"},
            {"spearman": "0.9", "top1_overlap": "0", "top3_overlap": "3", "top5_overlap": "5",
             "top3_ctx_overlap": "2", "top5_ctx_overlap": "4"}]
    assert summary_cells(rows) == [2, "0.70", "50%", "2.50", "4.00", "1.50", "3.00"]


def test_show_prints_counts_without_decimals():
    assert show("top_down_beats", 19.0) == "19"
    assert show("f_end", 0.12345) == "0.123"


def test_curves_from_csv_and_summaries(tmp_path):
    path = tmp_path / "removal.csv"
    lines = ["pair,role,order,k,f"]
    for order, values in {"top_down": [0.8, 0.0], "bottom_up": [0.8, 0.9],
                          "random_0": [0.8, 0.4], "random_1": [0.8, 0.5]}.items():
        lines += [f"1,hit,{order},{k},{v}" for k, v in enumerate(values)]
    path.write_text("\n".join(lines))
    curves = curves_from_csv(str(path))
    assert curves[("1", "hit")]["top_down"] == [0.8, 0.0]
    summary = removal_summaries(curves)[0]
    assert summary["f_full"] == 0.8
    assert summary["top_down_beats"] == 2       # both random curves have a higher mean score
    assert summary["bottom_up_beats"] == 2


def test_median_curves_takes_the_median_over_cases():
    curves = {("1", "hit"): {"top_down": [1.0, 0.0], "bottom_up": [1.0, 1.0], "random_0": [1.0, 0.5]},
              ("2", "hit"): {"top_down": [1.0, 0.2], "bottom_up": [1.0, 1.0], "random_0": [1.0, 0.7]},
              ("3", "hit"): {"top_down": [1.0, 0.4], "bottom_up": [1.0, 1.0], "random_0": [1.0, 0.9]},
              ("4", "miss"): {"top_down": [0.0, 0.0], "bottom_up": [0.0, 0.0], "random_0": [0.0, 0.0]}}
    result = median_curves(curves, "hit")
    assert np.allclose(result["top_down"], [1.0, 0.2])
    assert np.allclose(result["random (mean of 20)"], [1.0, 0.7])
