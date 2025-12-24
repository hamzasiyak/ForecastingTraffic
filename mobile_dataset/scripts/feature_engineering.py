# ============================================
# Feature Engineering for Milano CDR Internet Traffic
# - Input : mobile_dataset/data/processed/milano_internet_combined.csv
# - Output: mobile_dataset/data/processed/milano_features.csv
# - Per cell (square_id) time-series features without leakage
# ============================================

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "milano_internet_combined.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "milano_features.csv"

LAG_STEPS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLL_WINDOWS = [3, 6, 12, 24]
CRITICAL_LAGS = ["lag_1", "lag_24", "lag_168"]


def build_features(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not input_path.exists():
        raise SystemExit(f"❌ Girdi bulunamadı: {input_path}")

    df = pd.read_csv(input_path)
    if "time_interval" not in df.columns or "internet_traffic" not in df.columns or "square_id" not in df.columns:
        raise SystemExit("❌ Beklenen kolonlar bulunamadı (square_id, time_interval, internet_traffic).")

    df["time_interval"] = pd.to_datetime(df["time_interval"])
    df = df.sort_values(["square_id", "time_interval"]).reset_index(drop=True)
    rows_in = len(df)

    # --- Lag features ---
    for k in LAG_STEPS:
        df[f"lag_{k}"] = df.groupby("square_id")["internet_traffic"].shift(k)

    # --- Rolling stats (includes current point; no future info) ---
    traffic_by_cell = df.groupby("square_id")["internet_traffic"]
    for w in ROLL_WINDOWS:
        roll = traffic_by_cell.rolling(window=w, min_periods=1)
        df[f"rolling_mean_{w}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"rolling_std_{w}"] = roll.std().reset_index(level=0, drop=True)

    # --- Time encodings ---
    df["hour"] = df["time_interval"].dt.hour
    df["minute"] = df["time_interval"].dt.minute
    df["weekday"] = df["time_interval"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)
    # aliases for backward-compatibility with older training scripts
    df["day_of_week"] = df["weekday"]
    df["dow_sin"] = df["weekday_sin"]
    df["dow_cos"] = df["weekday_cos"]

    # --- Trend / difference features ---
    df["diff_1"] = df.groupby("square_id")["internet_traffic"].diff(1)
    df["diff_24"] = df.groupby("square_id")["internet_traffic"].diff(24)
    df["pct_change_1"] = df.groupby("square_id")["internet_traffic"].pct_change(1)
    df["pct_change_24"] = df.groupby("square_id")["internet_traffic"].pct_change(24)
    # pct_change, önceki değer 0 olduğunda ±inf üretebilir; onları düşür
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- Drop rows lacking critical history (skip impossible lags) ---
    available_critical = [c for c in CRITICAL_LAGS if df[c].notna().any()]
    if not available_critical:
        print("⚠️ Hiçbir kritik lag dolu değil; satır drop edilmeyecek (lag sütunları NaN kalacak).")
        mask = pd.Series(True, index=df.index)
    else:
        if len(available_critical) < len(CRITICAL_LAGS):
            missing = set(CRITICAL_LAGS) - set(available_critical)
            print(f"⚠️ Bazı kritik lag sütunları tamamen NaN: {sorted(missing)} — drop mask'inde kullanılmadı.")
        mask = df[available_critical].notna().all(axis=1)

    df_out = df.loc[mask].reset_index(drop=True)
    rows_out = len(df_out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    base_cols = {"square_id", "time_interval", "internet_traffic"}
    feature_cols = [c for c in df_out.columns if c not in base_cols]

    print("✅ Özellik üretimi tamamlandı.")
    print(f"   Girdi satır:  {rows_in:,}")
    print(f"   Çıktı satır:  {rows_out:,} (silinen: {rows_in - rows_out:,})")
    print(f"   Özellik sayısı: {len(feature_cols)}")
    print(f"   Çıktı: {output_path}")

    return df, df_out


def main():
    build_features(DEFAULT_INPUT, DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
