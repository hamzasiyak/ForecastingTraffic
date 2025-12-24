import argparse
import json
from pathlib import Path
import pandas as pd

def main(csv_path: Path):
    csv_path = csv_path.resolve()
    root = csv_path.parents[2] if csv_path.parts[-3:] == ("data","processed",csv_path.name) else Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] CSV: {csv_path}")
    if not csv_path.exists():
        print(f"[ERROR] Dosya bulunamadı: {csv_path}")
        return

    file_size_mb = csv_path.stat().st_size / (1024*1024)
    print(f"[INFO] Boyut: {file_size_mb:.2f} MB")

    # İlk satıra bakıp time sütununu tespit et
    sample = pd.read_csv(csv_path, nrows=5, low_memory=False)
    time_col = None
    for c in ["time_interval","timestamp","time","date","datetime"]:
        if c in sample.columns:
            time_col = c
            break

    parse_dates = [time_col] if time_col else None
    df = pd.read_csv(csv_path, low_memory=False, parse_dates=parse_dates)

    print(f"[INFO] Şekil: {df.shape[0]:,} satır x {df.shape[1]} kolon")
    print(f"[INFO] Kolonlar: {list(df.columns)}")

    # Belirgin ID / hedef kolonları
    id_col = "square_id" if "square_id" in df.columns else None
    target_candidates = ["internet_traffic","traffic","volume","target"]
    target = next((c for c in target_candidates if c in df.columns), None)

    # Dtype ve bellek
    mem = df.memory_usage(deep=True).sum()/(1024*1024)
    print(f"[INFO] Bellek kullanım tahmini: {mem:.2f} MB")
    print("[INFO] Dtype özet:")
    print(df.dtypes)

    # Zaman aralığı
    if time_col:
        tmin = df[time_col].min()
        tmax = df[time_col].max()
        print(f"[INFO] Tarih aralığı: {tmin} → {tmax}")
        # Yaklaşık örnekleme sıklığı
        try:
            freq = pd.infer_freq(df.sort_values(time_col)[time_col].dropna().unique())
            print(f"[INFO] Yaklaşık frekans: {freq}")
        except Exception:
            pass

    # Eksik değerler (ilk 10)
    na = df.isna().mean().sort_values(ascending=False)
    if na.gt(0).any():
        print("[INFO] Eksik oranları (ilk 10):")
        print((na[na>0].head(10)*100).round(2).astype(str) + "%")
    else:
        print("[INFO] Eksik değer yok.")

    # Duplikeler
    dup_count = 0
    if id_col and time_col:
        dup_count = df.duplicated(subset=[id_col, time_col]).sum()
        print(f"[INFO] Duplike (square_id, {time_col}) kayıt sayısı: {dup_count}")

    # Hedef istatistikleri
    if target:
        desc = df[target].describe(percentiles=[.05,.25,.5,.75,.95])
        print("[INFO] Hedef istatistikleri:")
        print(desc.to_string())

        # Hücre bazlı özet
        if id_col:
            per_cell = df.groupby(id_col)[target].agg(['count','mean','std','min','max']).reset_index()
            per_cell.to_csv(results_dir / "per_cell_traffic_summary.csv", index=False)
            print(f"[INFO] Hücre bazlı özet kaydedildi: {results_dir / 'per_cell_traffic_summary.csv'}")

    # Örnek kayıtlar
    df.head(100).to_csv(results_dir / "sample_rows.csv", index=False)
    print(f"[INFO] Örnek 100 satır: {results_dir / 'sample_rows.csv'}")

    # Profil özetini JSON olarak kaydet
    profile = {
        "path": str(csv_path),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(map(str, df.columns)),
        "time_col": time_col,
        "id_col": id_col,
        "target": target,
        "file_mb": round(file_size_mb,2),
        "mem_mb": round(mem,2),
        "duplicates": int(dup_count),
        "time_min": str(df[time_col].min()) if time_col else None,
        "time_max": str(df[time_col].max()) if time_col else None,
        "na_top10": (na[na>0].head(10).round(6).to_dict() if na.gt(0).any() else {}),
    }
    (results_dir / "data_profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    print(f"[INFO] Profil kaydedildi: {results_dir / 'data_profile.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(Path(__file__).resolve().parents[1] / "data" / "processed" / "milano_internet_combined.csv"))
    args = ap.parse_args()
    main(Path(args.csv))