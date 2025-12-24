# MobilCom Trafik Tahmin Projesi

Kasım 2013 Milano CDR verisiyle, proje önerisindeki amaçlara uygun kısa vadeli mobil internet trafik tahmini.

## Problem & Amaç (öneri ile hizalı)
- Trafik yükü saat ve hücreye göre değişiyor; kısa vadeli tahmin QoS ve kapasite planlama için kritik.
- Hedef: Ham CDR verisinden temiz veri üretip, zaman/lage bağlı özellikler çıkararak kısa vadeli trafik tahmini yapmak.
- Modeller: Baseline (Naive, Moving Average, ARIMA/Prophet), ML (RandomForest, XGBoost) ve DL (LSTM).
- Metrikler: MAE, RMSE, R²; çıktıların görsellerle kıyaslanması.

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

## Çalıştırma Adımları
```bash
# 1) Ham veriyi birleştir ve temizle
python mobile_dataset/scripts/prepare_data.py

# 2) Özellik üret
python mobile_dataset/scripts/feature_engineering.py

# 3a) XGBoost eğit (örnek)
python mobile_dataset/scripts/train_xgb.py

# 3b) LSTM (tek hücre)
python mobile_dataset/scripts/train_lstm.py --cell-id 1129

# 4) Tek hücrede değerlendirme (kayıtlı RF/XGB modeli)
python mobile_dataset/scripts/evaluate_ml_models.py --cell-id 1129 --model-path mobile_dataset/results/xgb_model.pkl

# 5) ARIMA/Prophet baselineleri
python mobile_dataset/scripts/baseline_time_models.py --cell-id 1129

# 6) Tüm çıktıların temiz özeti (tablo + grafik)
python mobile_dataset/scripts/summarize_results.py
```

## Çıktılar
- Metrix JSON: `mobile_dataset/results/metrics.json`
- Temiz özet: `mobile_dataset/results/summary.md`, CSV tablo: `summary_models.csv`, bar grafik: `model_rmse_bar.png`
- Örnek tahmin grafiklerini görmek için: `mobile_dataset/results/*pred*.png` ve `baseline_preds_cell_<id>.png`
- Sunum taslağı: `mobile_dataset/scripts/Mobile_Network_Traffic_Prediction_Presentation.pptx`

## Sonraki Adımlar
- Walk-forward backtest ve çoklu split ile sağlamlık analizi.
- En kötü/iyi hücre hatalarının dağılımı ve hata haritaları.
- Dış özellikler (hava, etkinlik, bölge tipi) ve mekansal komşuluk eklemek.
