"""
Train/evaluate classical time-series baselines (ARIMA & Prophet) on a single cell.

Usage:
    python baseline_time_models.py --cell-id 1129 --train-ratio 0.8
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:  # pragma: no cover
    SARIMAX = None

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover
    Prophet = None

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "milano_internet_combined.csv"
RESULTS_PATH = ROOT / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def add_time_regressors(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    hours = enriched["time_interval"].dt.hour
    weekday = enriched["time_interval"].dt.dayofweek
    enriched["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    enriched["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    enriched["dow_sin"] = np.sin(2 * np.pi * weekday / 7)
    enriched["dow_cos"] = np.cos(2 * np.pi * weekday / 7)
    return enriched


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_arima(train: pd.Series, test_horizon: int) -> Optional[np.ndarray]:
    if SARIMAX is None:
        print("⚠️ statsmodels (SARIMAX) bulunamadı. `pip install statsmodels` sonrası tekrar deneyin.")
        return None
    model = SARIMAX(
        train,
        order=(2, 0, 2),
        seasonal_order=(1, 0, 1, 24),  # capture daily seasonality
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=test_horizon)
    return forecast.to_numpy()


def run_prophet(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Optional[np.ndarray]:
    if Prophet is None:
        print("⚠️ Prophet paketi bulunamadı. `pip install prophet` sonrası tekrar deneyin.")
        return None
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        seasonality_mode=args.prophet_seasonality_mode,
        changepoint_prior_scale=args.prophet_changepoint_scale,
    )
    model.add_seasonality(
        name="daily",
        period=1,
        fourier_order=args.prophet_daily_fourier,
    )
    model.add_seasonality(
        name="weekly",
        period=7,
        fourier_order=args.prophet_weekly_fourier,
    )

    reg_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    for col in reg_cols:
        model.add_regressor(col, prior_scale=args.prophet_regressor_prior)

    train_enriched = add_time_regressors(train_df)
    future_enriched = add_time_regressors(future_df)
    model.fit(
        train_enriched.rename(columns={"time_interval": "ds", "internet_traffic": "y"})
    )
    forecast = model.predict(
        future_enriched.rename(columns={"time_interval": "ds"})
    )
    return forecast["yhat"].to_numpy()


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["time_interval"])
    cell_df = (
        df.loc[df["square_id"] == args.cell_id, ["time_interval", "internet_traffic"]]
        .sort_values("time_interval")
        .reset_index(drop=True)
    )

    if cell_df.empty:
        raise SystemExit(f"Cell {args.cell_id} için veri bulunamadı.")

    split_idx = int(len(cell_df) * args.train_ratio)
    train = cell_df.iloc[:split_idx]
    test = cell_df.iloc[split_idx:]

    if len(test) == 0:
        raise SystemExit("Test için yeterli veri bulunamadı; train_ratio değerini düşürün.")

    metrics: Dict[str, Dict[str, float]] = {}

    predictions: Dict[str, np.ndarray] = {}

    # ARIMA
    arima_pred = run_arima(train["internet_traffic"], len(test))
    if arima_pred is not None:
        metrics[f"arima_cell_{args.cell_id}"] = evaluate(
            test["internet_traffic"].to_numpy(), arima_pred
        )
        predictions["ARIMA"] = arima_pred

    # Prophet
    prophet_pred = run_prophet(train, test[["time_interval"]], args)
    if prophet_pred is not None:
        metrics[f"prophet_cell_{args.cell_id}"] = evaluate(
            test["internet_traffic"].to_numpy(), prophet_pred
        )
        predictions["Prophet"] = prophet_pred

    if not metrics:
        print("⚠️ Hiçbir baseline çalıştırılamadı. Gerekli bağımlılıkları yükledikten sonra tekrar deneyin.")
        return

    metrics_path = RESULTS_PATH / "metrics.json"
    existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    existing.update(metrics)
    metrics_path.write_text(json.dumps(existing, indent=2))

    print("✅ Baseline sonuçları güncellendi:")
    for name, vals in metrics.items():
        print(f"  {name}: {vals}")

    if (not args.no_plot) and predictions:
        plot_df = test.copy()
        plot_df = plot_df.rename(columns={"internet_traffic": "Actual"})
        for model_name, preds in predictions.items():
            plot_df[model_name] = preds

        plt.figure(figsize=(10, 4))
        plt.plot(plot_df["time_interval"], plot_df["Actual"], label="Actual", linewidth=2)
        for model_name in ["ARIMA", "Prophet"]:
            if model_name in plot_df.columns:
                plt.plot(
                    plot_df["time_interval"],
                    plot_df[model_name],
                    label=model_name,
                    linewidth=1.5,
                )
        plt.title(f"Cell {args.cell_id} Forecast vs Actual")
        plt.xlabel("Time")
        plt.ylabel("Internet Traffic")
        plt.legend()
        plt.tight_layout()
        out_path = RESULTS_PATH / f"baseline_preds_cell_{args.cell_id}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"📈 Grafik kaydedildi: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARIMA & Prophet baselines")
    parser.add_argument("--cell-id", type=int, default=1129, help="Hedef hücre ID")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/Test bölme oranı (0-1 arası)",
    )
    parser.add_argument(
        "--prophet-seasonality-mode",
        choices=["additive", "multiplicative"],
        default="additive",
    )
    parser.add_argument(
        "--prophet-changepoint-scale",
        type=float,
        default=0.1,
        help="Prophet changepoint_prior_scale değeri",
    )
    parser.add_argument(
        "--prophet-daily-fourier",
        type=int,
        default=10,
        help="Günlük seasonality için fourier order",
    )
    parser.add_argument(
        "--prophet-weekly-fourier",
        type=int,
        default=8,
        help="Haftalık seasonality için fourier order",
    )
    parser.add_argument(
        "--prophet-regressor-prior",
        type=float,
        default=0.5,
        help="Ek regresörler için prior scale",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Grafik üretimini kapat",
    )
    main(parser.parse_args())
