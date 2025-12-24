"""
Summarize training/evaluation outputs and create tidy artifacts for reporting.

Creates:
- results/summary.md : Markdown report with dataset stats and key metrics.
- results/summary_models.csv : Flattened metrics table for quick reuse.
- results/model_rmse_bar.png : Compact bar chart comparing RMSE scores.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = ROOT / "data" / "processed" / "milano_internet_combined.csv"
METRICS_PATH = ROOT / "results" / "metrics.json"
SUMMARY_MD = ROOT / "results" / "summary.md"
SUMMARY_CSV = ROOT / "results" / "summary_models.csv"
PLOT_PATH = ROOT / "results" / "model_rmse_bar.png"


@dataclass
class MetricEntry:
    model: str
    mae: float
    rmse: float
    r2: float
    details: Dict[str, str]


def load_dataset_stats() -> Optional[Dict[str, str]]:
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH, parse_dates=["time_interval"])
    if df.empty:
        return None
    return {
        "rows": f"{len(df):,}",
        "cells": f"{df['square_id'].nunique():,}",
        "start": str(df["time_interval"].min()),
        "end": str(df["time_interval"].max()),
    }


def load_metrics() -> Dict:
    if not METRICS_PATH.exists():
        raise SystemExit(f"❌ Metrics dosyası bulunamadı: {METRICS_PATH}")
    with METRICS_PATH.open() as f:
        return json.load(f)


def collect_prefixed(metrics: Dict, prefix: str, label: str) -> List[MetricEntry]:
    entries: List[MetricEntry] = []
    for key, val in metrics.items():
        if not (isinstance(key, str) and key.startswith(prefix) and isinstance(val, dict)):
            continue
        cell_id = key.removeprefix(prefix)
        mae, rmse, r2 = val.get("mae"), val.get("rmse"), val.get("r2")
        if mae is None or rmse is None or r2 is None:
            continue
        entries.append(
            MetricEntry(
                model=f"{label} (cell {cell_id})",
                mae=float(mae),
                rmse=float(rmse),
                r2=float(r2),
                details={"cell_id": cell_id},
            )
        )
    return entries


def pick_best(entries: Iterable[MetricEntry], minimize: str = "rmse") -> Optional[MetricEntry]:
    entries = list(entries)
    if not entries:
        return None
    return sorted(entries, key=lambda e: getattr(e, minimize))[0]


def build_tables(metrics: Dict) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    # Baselines
    baseline_rows: List[MetricEntry] = []
    naive = metrics.get("baseline_naive")
    if naive:
        baseline_rows.append(
            MetricEntry("Naive (t-1)", float(naive["mae"]), float(naive["rmse"]), float(naive["r2"]), {})
        )
    for k, v in metrics.get("baseline_ma", {}).items():
        baseline_rows.append(
            MetricEntry(f"MovingAvg {k.upper()}", float(v["mae"]), float(v["rmse"]), float(v["r2"]), {})
        )
    arima_best = pick_best(collect_prefixed(metrics, "arima_cell_", "ARIMA"))
    prophet_best = pick_best(collect_prefixed(metrics, "prophet_cell_", "Prophet"))
    if arima_best:
        baseline_rows.append(arima_best)
    if prophet_best:
        baseline_rows.append(prophet_best)
    if baseline_rows:
        tables["Baselines"] = pd.DataFrame([vars(r) for r in baseline_rows])

    # ML models
    ml_rows: List[MetricEntry] = []
    if {"rf_mae", "rf_rmse", "rf_r2"} <= metrics.keys():
        ml_rows.append(
            MetricEntry("RandomForest", float(metrics["rf_mae"]), float(metrics["rf_rmse"]), float(metrics["rf_r2"]), {})
        )
    if {"xgb_mae", "xgb_rmse", "xgb_r2"} <= metrics.keys():
        ml_rows.append(
            MetricEntry("XGBoost", float(metrics["xgb_mae"]), float(metrics["xgb_rmse"]), float(metrics["xgb_r2"]), {})
        )
    if ml_rows:
        tables["ML Models"] = pd.DataFrame([vars(r) for r in ml_rows])

    # Deep learning
    lstm_best = pick_best(collect_prefixed(metrics, "lstm_cell_", "LSTM"))
    if lstm_best:
        tables["Deep Learning"] = pd.DataFrame([vars(lstm_best)])

    return tables


def save_bar_plot(candidates: List[MetricEntry]) -> None:
    if not candidates:
        return
    models = [c.model for c in candidates]
    rmses = [c.rmse for c in candidates]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(models, rmses, color="#2e86de")
    plt.ylabel("RMSE")
    plt.title("Model RMSE Comparison")
    plt.xticks(rotation=20, ha="right")
    for bar, rmse in zip(bars, rmses):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{rmse:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def write_markdown(dataset_stats: Optional[Dict[str, str]], tables: Dict[str, pd.DataFrame]) -> None:
    def to_markdown(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        separator = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines = [header, separator]
        for _, row in df.iterrows():
            rendered = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    rendered.append(f"{val:.3f}")
                else:
                    rendered.append(str(val))
            lines.append("| " + " | ".join(rendered) + " |")
        return "\n".join(lines)

    lines: List[str] = ["# Mobile Network Traffic Prediction – Summary", ""]
    if dataset_stats:
        lines.append("## Dataset")
        lines.append(
            f"- Rows: {dataset_stats['rows']}, Cells: {dataset_stats['cells']}, "
            f"Range: {dataset_stats['start']} → {dataset_stats['end']}"
        )
        lines.append("")
    for name, df in tables.items():
        if df.empty:
            continue
        lines.append(f"## {name}")
        lines.append(to_markdown(df[["model", "mae", "rmse", "r2"]]))
        lines.append("")
    lines.append(f"Artifacts: `{SUMMARY_CSV.name}`, `{PLOT_PATH.name}`")
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("\n".join(lines))


def main() -> None:
    metrics = load_metrics()
    dataset_stats = load_dataset_stats()
    tables = build_tables(metrics)
    if not tables:
        raise SystemExit("❌ Özetlenecek metrik bulunamadı.")

    # Flatten and persist all model rows
    flat_rows: List[Dict[str, str]] = []
    for df in tables.values():
        flat_rows.extend(df.to_dict(orient="records"))
    pd.DataFrame(flat_rows).to_csv(SUMMARY_CSV, index=False)

    # Plot for the top representatives
    plot_candidates: List[MetricEntry] = []
    for name in ["Baselines", "ML Models", "Deep Learning"]:
        df = tables.get(name)
        if df is None or df.empty:
            continue
        row = df.iloc[df["rmse"].astype(float).argmin()]
        plot_candidates.append(
            MetricEntry(
                model=str(row["model"]),
                mae=float(row["mae"]),
                rmse=float(row["rmse"]),
                r2=float(row["r2"]),
                details=row.get("details", {}) if isinstance(row.get("details"), dict) else {},
            )
        )
    save_bar_plot(plot_candidates)
    write_markdown(dataset_stats, tables)
    print(f"✅ Özet oluşturuldu: {SUMMARY_MD}")
    print(f"📄 Tablo: {SUMMARY_CSV}")
    if PLOT_PATH.exists():
        print(f"📈 Grafik: {PLOT_PATH}")


if __name__ == "__main__":
    main()
