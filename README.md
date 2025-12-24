# MobilCom Trafik Tahmin Projesi

Milano 2013 CDR verisiyle kısa vadeli mobil internet trafik tahmini - Makine öğrenmesi ile 10,000+ hücre analizi.

## 🎯 Proje Özeti
- **Veri:** Telecom Italia Milano CDR (2013-11-01 → 11-07), 10 dakikalık aralıklar, 10,000+ hücre
- **Problem:** Hücre bazlı internet trafiğini bir sonraki zaman dilimi için tahmin etme
- **Çözüm:** Feature engineering (29 özellik: lag, rolling stats, time features) + ML modelleri
- **Sonuç:** RandomForest R²=0.998 (MAE=4.99), XGBoost R²=0.987 (MAE=10.87)

## 📊 Ana Analiz
**[end_to_end.ipynb](mobile_dataset/scripts/end_to_end.ipynb)** - Comprehensive ML Pipeline
- Data preprocessing ve feature engineering
- Baseline modeller (Naive, Moving Average)
- ML modeller (RandomForest, XGBoost)
- Feature importance, correlation, residual analysis
- Cross-validation (TimeSeriesSplit 5-fold)
- Hyperparameter tuning (RandomizedSearchCV)
- Model comparison ve forecast visualization

> 💡 **Hızlı Başlangıç:** Notebook'u açıp "Restart Kernel & Run All" yapın (~3-4 dk)

## 🛠️ Kurulum
```bash
# Virtual environment oluştur
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# Gereksinimleri yükle
pip install -r requirements.txt
```

## 📈 Modeller ve Performans
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **RandomForest** | **4.99** | **35.36** | **0.998** |
| XGBoost | 10.87 | 101.28 | 0.987 |
| Tuned RF | 9.15 | 79.34 | 0.992 |
| Moving Avg (3) | 52.50 | 148.37 | 0.972 |
| Naive (lag_1) | 63.22 | 173.81 | 0.961 |

## 🔍 Özellikler (29 total)
- **Lag features:** 1, 2, 3, 6, 12, 24 saatlik gecikmeler
- **Rolling stats:** 3h, 6h, 12h, 24h (mean, std, min, max)
- **Time features:** hour_sin/cos, dayofweek_sin/cos, is_weekend
- **Trend features:** traffic_diff, traffic_pct_change

## Veri Seti
- Kaynak: Telecom Italia Milano CDR (2013-11-01…07), 10 dakikalık aralıklar.
- Ham: `mobile_dataset/data/raw/sms-call-internet-mi-2013-11-*.csv`
- Temiz: `mobile_dataset/data/processed/milano_internet_combined.csv`
- Özellikli: `mobile_dataset/data/processed/milano_features.csv`
- Hedef: `internet_traffic` | Kimlik: `square_id` | Zaman: `time_interval`

## İş Akışı
1) **Temizleme:** Günlük CSV’leri birleştir, kolonları sabitle, duplikeleri topla (`prepare_data.py`).
2) **Özellikler:** Lag, rolling, saat/gün çevrimsel kodlama, fark/pct değişim (`feature_engineering.py`).
3) **Modeller:**  
   - Baseline: Naive & moving average; tek hücre için ARIMA/Prophet (`baseline_time_models.py`).  
   - ML: RandomForest/XGBoost tablo modelleri (`train_xgb.py`, RF modeli `evaluate_ml_models.py` ile yükleniyor).  
   - DL: Sekans-to-one LSTM tek hücre ya da otomatik hücre araması (`train_lstm.py`).
4) **Değerlendirme:** `metrics.json` güncellenir, tahmin vs gerçek görselleri `results/` altına yazılır.
5) **Özetleme:** Tüm metrikleri temiz tablo ve grafikle sun (`summarize_results.py`).

## 🚀 Hızlı Başlangıç

### Jupyter Notebook (Önerilen)
```bash
# Notebook'u aç ve tüm hücreleri çalıştır
jupyter notebook mobile_dataset/scripts/end_to_end.ipynb
# Veya VS Code'da "Restart Kernel & Run All"
```

### Script-Based Workflow (Opsiyonel)
```bash
# 1) Veri hazırlama
python mobile_dataset/scripts/prepare_data.py
python mobile_dataset/scripts/feature_engineering.py

# 2) Model eğitimi
python mobile_dataset/scripts/train_xgb.py

# 3) LSTM (tek hücre için)
python mobile_dataset/scripts/train_lstm.py --cell-id 1129

# 4) Sonuçları özetle
python mobile_dataset/scripts/summarize_results.py
```

## 📁 Proje Yapısı
```
MobilCom/
├── README.md
├── requirements.txt
├── .gitignore
└── mobile_dataset/
    ├── data/
    │   ├── raw/           # Ham CDR dosyaları (GitHub'da ignore)
    │   └── processed/     # İşlenmiş veriler (GitHub'da ignore)
    ├── scripts/
    │   ├── end_to_end.ipynb          # 🌟 ANA ANALİZ
    │   ├── prepare_data.py
    │   ├── feature_engineering.py
    │   ├── train_xgb.py
    │   ├── train_lstm.py
    │   └── ...
    └── results/           # Model outputs, metrics (GitHub'da ignore)
```

## 🎓 Metodoloji
1. **Data Preprocessing:** 7 günlük ham CSV → birleştirilmiş temiz data (1.44M satır)
2. **Feature Engineering:** 29 özellik (lag, rolling, time, trend features)
3. **Train/Test Split:** Time-based 80/20 split (temporal leakage önleme)
4. **Baseline Models:** Naive (lag_1), Moving Average (3/5/7)
5. **ML Models:** RandomForest, XGBoost with hyperparameter tuning
6. **Validation:** TimeSeriesSplit 5-fold cross-validation
7. **Evaluation:** MAE, RMSE, R² + görsel analizler

## 📊 Çıktılar
- **Metrics:** `mobile_dataset/results/metrics.json`
- **Görseller:** Feature importance, correlation heatmap, residual plots, forecast comparison
- **Model Dosyaları:** `xgb_model.json`, `lstm_model.pt` (eğer eğitildiyse)

## 💡 Notlar
- **Deep Learning:** LSTM notebook'tan çıkarıldı (10K hücre için per-cell eğitim 30+ saat). Alternatif: `train_lstm.py` script'i ile tek hücre analizi.
- **Performance:** MacBook optimize edildi - RandomizedSearchCV, subsampling, CPU-only execution.
- **Data Size:** Ham data dosyaları (~100MB+) `.gitignore` ile exclude edilmiş.

## 🔮 Gelecek Geliştirmeler
- Spatial features (komşu hücre trafiği)
- External features (hava durumu, events)
- Multi-step ahead forecasting
- Real-time inference pipeline
