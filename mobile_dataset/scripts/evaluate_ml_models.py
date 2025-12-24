"""
Evaluate a saved ML model (e.g., RandomForest/XGBoost) on a single cell.

Example:
    python evaluate_ml_models.py --cell-id 1129 \
        --model-path ../results/rf_model.pkl
"""

import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_FILE = ROOT / "data" / "processed" / "milano_features.csv"
DEFAULT_MODEL = ROOT / "results" / "rf_model.pkl"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS: List[str] = [
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_12",
    "rolling_mean_24",
    "rolling_std_3",
    "rolling_std_6",
    "rolling_std_12",
    "rolling_std_24",
    "diff_1",
    "diff_24",
    "pct_change_1",
    "pct_change_24",
]


def main(args: argparse.Namespace) -> None:
    features_path = Path(args.features_file or DEFAULT_FEATURE_FILE)
    model_path = Path(args.model_path or DEFAULT_MODEL)

    if not features_path.exists():
        raise SystemExit(f"Özellik dosyası bulunamadı: {features_path}")
    if not model_path.exists():
        raise SystemExit(f"Model dosyası bulunamadı: {model_path}")

    df = pd.read_csv(features_path, parse_dates=["time_interval"])
    missing_cols = [c for c in FEATURE_COLUMNS + ["square_id", "internet_traffic"] if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Veride eksik kolonlar var: {missing_cols}")

    cell_df = df.loc[df["square_id"] == args.cell_id].dropna(subset=FEATURE_COLUMNS + ["internet_traffic"]).copy()
    if cell_df.empty:
        raise SystemExit(f"Cell {args.cell_id} için veri bulunamadı.")

    model = joblib.load(model_path)
    X = cell_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = cell_df["internet_traffic"].to_numpy(dtype=float)
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    print(f"✅ Model değerlendirmesi ({model.__class__.__name__}) - Cell {args.cell_id}")
    print(f"   MAE : {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²  : {r2:.4f}")

    if not args.no_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(cell_df["time_interval"], y, label="Actual", linewidth=2)
        plt.plot(cell_df["time_interval"], preds, label="Prediction", linewidth=1.5)
        plt.title(f"{model.__class__.__name__} vs Actual (Cell {args.cell_id})")
        plt.xlabel("Time")
        plt.ylabel("Internet Traffic")
        plt.legend()
        plt.tight_layout()
        out_path = RESULTS_DIR / f"{model.__class__.__name__.lower()}_cell_{args.cell_id}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"📈 Grafik kaydedildi: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved ML model on a specific cell.")
    parser.add_argument("--cell-id", type=int, required=True, help="Değerlendirilecek square_id")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--features-file", type=str, default=str(DEFAULT_FEATURE_FILE))
    parser.add_argument("--no-plot", action="store_true", help="Grafik üretme")
    main(parser.parse_args())
