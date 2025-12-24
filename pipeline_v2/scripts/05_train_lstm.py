"""
Wrapper: Tek hücre veya taramalı LSTM eğitimi.
Delegates to: mobile_dataset/scripts/train_lstm.py
"""

import argparse
import subprocess

from common_paths import OLD_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM on a specific cell or search.")
    parser.add_argument("--cell-id", type=int, help="Belirli hücre ID'si.")
    parser.add_argument("--search-cells", type=str, help="Örn. '1129,2552'")
    parser.add_argument("--max-search-cells", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    script = OLD_ROOT / "scripts" / "train_lstm.py"
    cmd = ["python3", str(script)]
    if args.cell_id is not None:
        cmd += ["--cell-id", str(args.cell_id)]
    if args.search_cells:
        cmd += ["--search-cells", args.search_cells]
    if args.max_search_cells:
        cmd += ["--max-search-cells", str(args.max_search_cells)]
    if args.no_plot:
        cmd.append("--no-plot")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
