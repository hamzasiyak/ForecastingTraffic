"""
Wrapper: Kayıtlı ML modelini belirli bir hücrede değerlendirir.
Delegates to: mobile_dataset/scripts/evaluate_ml_models.py
"""

import argparse
import subprocess
from pathlib import Path

from common_paths import OLD_ROOT, MODELS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved ML model on a cell.")
    parser.add_argument("--cell-id", type=int, required=True)
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "xgb_model.pkl"))
    parser.add_argument("--features-file", type=str, default=str(OLD_ROOT / "data" / "processed" / "milano_features.csv"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    script = OLD_ROOT / "scripts" / "evaluate_ml_models.py"
    cmd = [
        "python3",
        str(script),
        "--cell-id",
        str(args.cell_id),
        "--model-path",
        args.model,
        "--features-file",
        args.features_file,
    ]
    if args.no_plot:
        cmd.append("--no-plot")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
