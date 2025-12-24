# ============================================
# Milano CDR Internet Traffic Preparer (Fixed Columns)
# ============================================

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve project root (this file lives in ./scripts)
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw"   # CSV'ler project root altında ./data/raw
PROCESSED_PATH = ROOT / "data" / "processed"
RESULTS_PATH = ROOT / "results"

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# ------------------------------
# 1. Dosyaları oku ve birleştir
# ------------------------------
files = glob.glob(str(RAW_PATH / "sms-call-internet-mi-2013-11-*.csv"))
print(f"{len(files)} adet dosya bulundu. (aranan dizin: {RAW_PATH})")

if not files:
    print("❌ Hiç CSV bulunamadı. Lütfen dosyaların şu dizinde olduğundan emin olun:")
    print(f"   {RAW_PATH}")
    print("   ve isim deseninin 'sms-call-internet-mi-2013-11-*.csv' ile eşleştiğini kontrol edin.")
    raise SystemExit(1)

dfs = []
for f in sorted(files):
    print(f"→ {os.path.basename(f)} okunuyor...")
    data = pd.read_csv(f)
    data.columns = [c.strip().lower() for c in data.columns]
    # Gereken sütunları seç
    df_part = data[['datetime', 'cellid', 'internet']].copy()
    df_part.rename(columns={
        'datetime': 'time_interval',
        'cellid': 'square_id',
        'internet': 'internet_traffic'
    }, inplace=True)
    dfs.append(df_part)

# ------------------------------
# 2. Tek veri seti haline getir
# ------------------------------
df = pd.concat(dfs, ignore_index=True)

# ------------------------------
# 2.1 Temizlik: tipler, eksikler, çoğulları birleştirme
# ------------------------------
# internet_traffic sayısal olsun, eksikler 0 kabul edilsin (toplam trafik)
df['internet_traffic'] = pd.to_numeric(df['internet_traffic'], errors='coerce').fillna(0.0)

# Aynı (square_id, time_interval) için birden çok satır varsa toplayarak tekilleştir
before_rows = len(df)
dup_keys = ['time_interval', 'square_id']
dup_count = df.duplicated(subset=dup_keys, keep=False).sum()
print(f"\nBilgi: Tekilleştirme öncesi satır: {before_rows:,}, duplike kayıt: {dup_count:,}")

df = (df
    .groupby(dup_keys, as_index=False, sort=False)['internet_traffic']
    .sum()
)
after_rows = len(df)
print(f"Bilgi: Tekilleştirme sonrası satır: {after_rows:,} (azalan: {before_rows - after_rows:,})")
df['time_interval'] = pd.to_datetime(df['time_interval'])

print("\n✅ Veri başarıyla birleştirildi:")
print(df.head())

# ------------------------------
# 3. Ortalama trafik hesapla
# ------------------------------
df_mean = df.groupby('time_interval')['internet_traffic'].mean().reset_index()

# ------------------------------
# 4. Görselleştir
# ------------------------------
plt.figure(figsize=(10,4))
plt.plot(df_mean['time_interval'], df_mean['internet_traffic'], color='orange')
plt.title("Average Internet Traffic Across All Cells")
plt.xlabel("Time")
plt.ylabel("Average Traffic Load")
plt.grid(True)
plt.tight_layout()
plt.savefig(str(RESULTS_PATH / "average_traffic.png"))
plt.close()

print("\n✅ Ortalama trafik grafiği kaydedildi (results/average_traffic.png).")

# ------------------------------
# 5. Veriyi kaydet
# ------------------------------
processed_file = PROCESSED_PATH / "milano_internet_combined.csv"
print("\n💾 CSV yazılıyor (bu işlem birkaç dakika sürebilir)...")
df.to_csv(processed_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
print(f"✅ Temiz veri kaydedildi: {processed_file}")
