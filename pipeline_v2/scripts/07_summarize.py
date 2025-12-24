"""
Wrapper: metrics özetlerini ve grafiklerini üretir.
Delegates to: mobile_dataset/scripts/summarize_results.py
"""

import subprocess

from common_paths import OLD_ROOT


def main() -> None:
    script = OLD_ROOT / "scripts" / "summarize_results.py"
    subprocess.run(["python3", str(script)], check=True)


if __name__ == "__main__":
    main()
