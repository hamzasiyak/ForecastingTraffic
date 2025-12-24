# ============================================
# Feature Engineering for Milano CDR Internet Traffic
# Input : ../data/processed/milano_internet_combined.csv
# Output: ../data/processed/milano_features.csv
# ============================================

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta

INPUT_FILE  = "./data/processed/milano_internet_combined.csv"
OUTPUT_FILE = "./data/processed/milano_features.csv"

class FeatureEngineering:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        # Ensure datetime column is properly converted
        try:
            if isinstance(self.df['datetime'].iloc[0], str):
                self.df['datetime'] = pd.to_datetime(
                    self.df['datetime'],
                    format='%Y-%m-%d %H:%M:%S'
                )
            print(f"✅ Successfully converted datetime column. Sample: {self.df['datetime'].iloc[0]}")
        except Exception as e:
            print(f"❌ Error converting datetime: {e}")
            print(f"Column type: {self.df['datetime'].dtype}")
            print(f"First few values: \n{self.df['datetime'].head()}")
            raise
        
        # Sort and reset index
        self.df.sort_values(['CellID', 'datetime'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def create_lag_features(self, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Her hücre için geçmiş zaman değerlerini özellik olarak ekler
        """
        for col in columns:
            for lag in lags:
                self.df[f'{col}_lag_{lag}'] = self.df.groupby('CellID')[col].shift(lag)
        return self.df

    def create_rolling_features(self, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """
        Hareketli ortalama ve standart sapma özellikleri oluşturur
        """
        for col in columns:
            for window in windows:
                # Hareketli ortalama
                self.df[f'{col}_rolling_mean_{window}'] = (
                    self.df.groupby('CellID')[col]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                # Hareketli standart sapma
                self.df[f'{col}_rolling_std_{window}'] = (
                    self.df.groupby('CellID')[col]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(0, drop=True)
                )
        return self.df

    def create_time_features(self) -> pd.DataFrame:
        """
        Gelişmiş zaman özellikleri oluşturur
        """
        # Temel zaman özellikleri
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['day'] = self.df['datetime'].dt.day
        self.df['weekday'] = self.df['datetime'].dt.weekday
        self.df['month'] = self.df['datetime'].dt.month
        
        # İş günü ve tatil özellikleri
        self.df['is_weekend'] = self.df['weekday'].isin([5, 6]).astype(int)
        self.df['is_business_hour'] = (
            (self.df['hour'].between(9, 17)) & 
            (~self.df['is_weekend'])
        ).astype(int)
        
        # Günün bölümleri
        self.df['day_part'] = pd.cut(
            self.df['hour'],
            bins=[-1, 6, 12, 18, 23],
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        
        # Cyclical encoding for time features
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour']/24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour']/24)
        
        return self.df

    def create_cell_features(self) -> pd.DataFrame:
        """
        Hücre bazlı özellikler oluşturur
        """
        # Hücre bazlı ortalama ve standart sapma
        for col in ['internet', 'smsin', 'smsout', 'callin', 'callout']:
            self.df[f'{col}_cell_mean'] = self.df.groupby('CellID')[col].transform('mean')
            self.df[f'{col}_cell_std'] = self.df.groupby('CellID')[col].transform('std')
        
        # Hücre bazlı yoğunluk özellikleri
        self.df['total_activity'] = self.df[['internet', 'smsin', 'smsout', 'callin', 'callout']].sum(axis=1)
        self.df['cell_activity_rank'] = self.df.groupby('CellID')['total_activity'].rank(pct=True)
        
        return self.df

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Girdi bulunamadı: {INPUT_FILE}")
        print("Önce prepare_data.py çalıştırıp birleşik veriyi üretmelisin.")
        sys.exit(1)

    # 1) veriyi oku
    df = pd.read_csv(INPUT_FILE)
    # beklenen kolonlar
    expected = {"time_interval", "square_id", "internet_traffic"}
    missing = expected - set(df.columns)
    if missing:
        print(f"❌ Beklenen sütun(lar) eksik: {missing}")
        sys.exit(1)

    # 2) tip dönüşümleri
    original_times = df["time_interval"].copy()
    # temizle gizli karakterleri ve boşlukları
    cleaned = original_times.astype(str).str.replace("\x00", "", regex=False).str.replace("\ufeff", "", regex=False).str.strip()
    # YYYY-MM-DD veya YYYY-MM-DD HH:MM:SS yakala
    extracted = cleaned.str.extract(r'(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?)')[0].str.replace("T", " ").str.strip()
    parsed = pd.to_datetime(extracted, errors="coerce")
    # tarih-only durumları için saat ekleyip tekrar dene
    n_bad = parsed.isna().sum()
    if n_bad:
        only_date_mask = extracted.str.match(r'^\d{4}-\d{2}-\d{2}$') & parsed.isna()
        if only_date_mask.any():
            parsed.loc[only_date_mask] = pd.to_datetime(extracted[only_date_mask] + " 00:00:00", errors="coerce")
        n_bad = parsed.isna().sum()
    if n_bad:
        print(f"❌ {n_bad} satır time_interval'a dönüştürülemedi. Örnekler (repr):")
        bad_idx = parsed.isna().to_numpy().nonzero()[0][:5]
        for i in bad_idx:
            print(repr(original_times.iloc[i]))
        sys.exit(1)
    df["time_interval"] = parsed
    df = df.sort_values(["square_id", "time_interval"]).reset_index(drop=True)

    # 3) zaman tabanlı özellikler
    df["hour"]         = df["time_interval"].dt.hour
    df["minute"]       = df["time_interval"].dt.minute
    df["day_of_week"]  = df["time_interval"].dt.dayofweek    # 0=Mon
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)

    # döngüsel kodlama (hour ve day_of_week)
    # 24 saatlik periyot
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    # 7 günlük periyot
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # 4) gecikmeli (lag) ve hareketli (rolling) özellikler
    # not: grup bazında (her square_id için) hesaplanır
    def add_lag_roll(g):
        g = g.copy()
        g["lag_1"]  = g["internet_traffic"].shift(1)     # 1 saat önce
        g["lag_2"]  = g["internet_traffic"].shift(2)     # 2 saat önce
        g["lag_3"]  = g["internet_traffic"].shift(3)     # 3 saat önce

        # KAÇAK ÖNLEME: Rolling istatistikleri 1 adım kaydırarak sadece geçmişi kullandır
        g["roll_mean_3"]  = (
            g["internet_traffic"].rolling(window=3, min_periods=1).mean().shift(1)   # son 3 saat
        )
        g["roll_mean_6"]  = (
            g["internet_traffic"].rolling(window=6, min_periods=1).mean().shift(1)   # son 6 saat
        )
        g["roll_mean_12"] = (
            g["internet_traffic"].rolling(window=12, min_periods=1).mean().shift(1)  # son 12 saat
        )
        g["roll_std_6"]   = (
            g["internet_traffic"].rolling(window=6, min_periods=1).std().shift(1).fillna(0.0)
        )
        return g

    df = df.groupby("square_id", group_keys=False).apply(add_lag_roll)

    # 5) hedef & özellik ayıklama için küçük temizlik
    # çok erken satırlarda lag'ler NaN olabilir; model eğitiminde ya atarız ya da doldururuz.
    # burada eğitimden önce düşürmek daha güvenli:
    before = len(df)
    df_clean = df.dropna(subset=["lag_1", "lag_2", "lag_3"]).reset_index(drop=True)
    after = len(df_clean)

    # 6) kaydet
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_clean.to_csv(OUTPUT_FILE, index=False)

    # 7) kısa özet
    n_cells  = df_clean["square_id"].nunique()
    t_start  = df_clean["time_interval"].min()
    t_end    = df_clean["time_interval"].max()
    print("✅ Özellikli veri hazırlandı.")
    print(f"   Kayıt sayısı (temiz): {after:,}  (silinen: {before - after:,})")
    print(f"   Hücre (cell) sayısı : {n_cells}")
    print(f"   Zaman aralığı       : {t_start}  →  {t_end}")
    print(f"   Çıktı               : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()