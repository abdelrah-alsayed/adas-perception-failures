"""One command for every table and figure: python -m explain_adas.make_report

It reads only results/ (and the KITTI pictures for the maps, if they are on disk). No GPU needed.
"""
import time

from explain_adas.report_figures import make_figures
from explain_adas.report_tables import make_tables


def main():
    start = time.time()
    tables = make_tables()
    print(f"tables: {len(tables)} written to results/tables/ (tables.md has all of them)")
    files = make_figures()
    print(f"figures: {len(files)} written (plain charts in results/figures/, maps in results/maps/)")
    print(f"done in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
