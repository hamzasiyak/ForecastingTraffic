"""
Train an XGBoost regressor on engineered Milano features and persist the model.

Example:
    python train_xgb.py --output ../results/xgb_model.pkl
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

FEATURES = [
    # time encodings
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # lags
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",
    # rolling stats
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_12",
    "rolling_mean_24",
    "rolling_std_3",
    "rolling_std_6",
    "rolling_std_12",
    "rolling_std_24",
    # trend/diff
    "diff_1",
    "diff_24",
    "pct_change_1",
    "pct_change_24",
]


def main(args: argparse.Namespace) -> None:
    features_path = Path(args.features_file).resolve()
    if not features_path.exists():
        raise SystemExit(f"Özellik dosyası bulunamadı: {features_path}")

    df = pd.read_csv(features_path, parse_dates=["time_interval"])
    missing = [c for c in FEATURES + ["internet_traffic"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Veride eksik kolonlar var: {missing}")

    df = df.sort_values("time_interval").reset_index(drop=True)
    split_idx = int(len(df) * args.train_ratio)
    if split_idx == 0 or split_idx >= len(df):
        raise SystemExit("Train/Test bölme oranı uygun değil.")

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(train_df[FEATURES], train_df["internet_traffic"])

    preds = model.predict(test_df[FEATURES])
    mae = mean_absolute_error(test_df["internet_traffic"], preds)
    rmse = np.sqrt(mean_squared_error(test_df["internet_traffic"], preds))
    r2 = r2_score(test_df["internet_traffic"], preds)

    print("✅ XGBoost eğitimi tamamlandı.")
    print(f"   Test MAE : {mae:.4f}")
    print(f"   Test RMSE: {rmse:.4f}")
    print(f"   Test R²  : {r2:.4f}")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"💾 Model kaydedildi: {output_path}")

    metrics = {
        "xgb_test_mae": mae,
        "xgb_test_rmse": rmse,
        "xgb_test_r2": r2,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": FEATURES,
    }
    metrics_path = output_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"📄 Test metrikleri: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost regressor on Milano features.")
    parser.add_argument(
        "--features-file",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "processed" / "milano_features.csv"),
    )
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parents[1] / "results" / "xgb_model.pkl"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    main(parser.parse_args())
