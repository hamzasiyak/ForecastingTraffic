from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT.parents[0] / "mobile_dataset"

# Varsayılan yollar (mevcut dataset yapısına göre)
RAW_DIR = OLD_ROOT / "data" / "raw"
PROCESSED_DIR = OLD_ROOT / "data" / "processed"
RESULTS_DIR = OLD_ROOT / "results"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

MPL_CACHE = REPORTS_DIR / ".matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
