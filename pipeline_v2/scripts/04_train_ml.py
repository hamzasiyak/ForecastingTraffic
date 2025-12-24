"""
Wrapper: RF/XGB modellerini eğitir (varsayılan parametrelerle).
Delegates to: mobile_dataset/scripts/train_xgb.py (XGB)
"""

import argparse
import subprocess
from pathlib import Path

from common_paths import OLD_ROOT, MODELS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML models (XGB for now).")
    parser.add_argument("--features-file", type=str, default=str(OLD_ROOT / "data" / "processed" / "milano_features.csv"))
    parser.add_argument("--output", type=str, default=str(MODELS_DIR / "xgb_model.pkl"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    script = OLD_ROOT / "scripts" / "train_xgb.py"
    cmd = [
        "python3",
        str(script),
        "--features-file",
        args.features_file,
        "--output",
        args.output,
        "--train-ratio",
        str(args.train_ratio),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
