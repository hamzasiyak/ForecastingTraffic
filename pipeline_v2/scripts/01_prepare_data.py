"""
Ham CSV'leri hafızayı şişirmeden birleştirir, temizler ve çıktı yazar.
- Girdi: mobile_dataset/data/raw/sms-call-internet-mi-2013-11-*.csv
- Çıktı: mobile_dataset/data/processed/milano_internet_combined.csv
- Grafik: mobile_dataset/results/average_traffic.png
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import pandas as pd

from common_paths import OLD_ROOT, PROCESSED_DIR, RAW_DIR, RESULTS_DIR, MPL_CACHE

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    files = sorted(RAW_DIR.glob("sms-call-internet-mi-2013-11-*.csv"))
    print(f"{len(files)} adet dosya bulundu. (aranan dizin: {RAW_DIR})")
    if not files:
        raise SystemExit("❌ Ham CSV bulunamadı.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = PROCESSED_DIR / "milano_internet_combined.csv"
    if out_file.exists():
        out_file.unlink()

    time_sum: Dict[str, float] = defaultdict(float)
    time_count: Dict[str, int] = defaultdict(int)
    total_rows = 0

    for idx, fpath in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {fpath.name} okunuyor...")
        df = pd.read_csv(
            fpath,
            usecols=["datetime", "CellID", "internet"],
        )
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"datetime": "time_interval", "cellid": "square_id", "internet": "internet_traffic"})
        df["internet_traffic"] = pd.to_numeric(df["internet_traffic"], errors="coerce").fillna(0.0)

        before = len(df)
        df = df.groupby(["time_interval", "square_id"], as_index=False)["internet_traffic"].sum()
        after = len(df)
        total_rows += after
        print(f"   satır: {before:,} -> tekilleşmiş: {after:,}")

        # Günlük ortalama için topla
        for t, val in zip(df["time_interval"], df["internet_traffic"]):
            time_sum[t] += float(val)
            time_count[t] += 1

        mode = "w" if idx == 1 else "a"
        header = idx == 1
        df.to_csv(out_file, mode=mode, header=header, index=False, date_format="%Y-%m-%d %H:%M:%S")

    print(f"\n✅ Temiz veri kaydedildi: {out_file} (toplam satır: {total_rows:,})")

    # Ortalama trafik grafiği (tüm hücreler)
    mean_records = []
    for t, s in time_sum.items():
        c = time_count[t]
        if c > 0:
            mean_records.append((t, s / c))
    mean_df = pd.DataFrame(mean_records, columns=["time_interval", "internet_traffic"])
    mean_df["time_interval"] = pd.to_datetime(mean_df["time_interval"])
    mean_df = mean_df.sort_values("time_interval")

    plt.figure(figsize=(10, 4))
    plt.plot(mean_df["time_interval"], mean_df["internet_traffic"], color="orange")
    plt.title("Average Internet Traffic Across All Cells")
    plt.xlabel("Time")
    plt.ylabel("Average Traffic Load")
    plt.grid(True)
    plt.tight_layout()
    avg_path = RESULTS_DIR / "average_traffic.png"
    plt.savefig(avg_path, dpi=150)
    plt.close()
    print(f"📈 Ortalama trafik grafiği kaydedildi: {avg_path}")


if __name__ == "__main__":
    main()
