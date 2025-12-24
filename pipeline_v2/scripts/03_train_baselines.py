"""
Wrapper: ARIMA/Prophet ve basit baselineleri tek hücrede çalıştırır.
Delegates to: mobile_dataset/scripts/baseline_time_models.py
"""

import argparse
import subprocess

from common_paths import OLD_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARIMA/Prophet baselines for a cell.")
    parser.add_argument("--cell-id", type=int, default=1129)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    script = OLD_ROOT / "scripts" / "baseline_time_models.py"
    cmd = [
        "python3",
        str(script),
        "--cell-id",
        str(args.cell_id),
        "--train-ratio",
        str(args.train_ratio),
    ]
    if args.no_plot:
        cmd.append("--no-plot")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
