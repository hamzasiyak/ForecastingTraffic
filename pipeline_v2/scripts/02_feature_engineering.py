"""
Wrapper: temiz veriden özellik üretir.
Delegates to: mobile_dataset/scripts/feature_engineering.py
"""

import subprocess
from pathlib import Path

from common_paths import OLD_ROOT


def main() -> None:
    script = OLD_ROOT / "scripts" / "feature_engineering.py"
    subprocess.run(["python3", str(script)], check=True)


if __name__ == "__main__":
    main()
