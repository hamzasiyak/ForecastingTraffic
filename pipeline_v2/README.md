# MobilCom Pipeline v2

Öneri ile uyumlu, sıralı çalışabilir yeni akış. Ham veri aynı yerde kalır (`mobile_dataset/data/raw`); bu klasör sadece wrapper ve rapor dosyalarını içerir.

## Dizinler
- `scripts/` : 01–07 adım scriptleri (mevcut kodları çağırır).
- `data/` : Girdi/çıktı için mantıksal kök (ham dosyalar `mobile_dataset/data/raw` içinde kalır).
- `models/` : Eğitilen model dosyaları (RF/XGB/LSTM).
- `reports/` : Özetler ve grafikler (`reports/plots/`).

## Çalıştırma Sırası
```bash
python scripts/01_prepare_data.py
python scripts/02_feature_engineering.py
python scripts/03_train_baselines.py --cell-id 1129
python scripts/04_train_ml.py
python scripts/05_train_lstm.py --cell-id 1129
python scripts/06_evaluate.py --cell-id 1129 --model rf.pkl
python scripts/07_summarize.py
```

## Notlar
- Ham veriyi kopyalamıyoruz; `mobile_dataset/data/raw` altında kalıyor.
- Matplotlib cache proje içine yönlendirilir (`reports/.matplotlib`).
- Tüm metrikler `mobile_dataset/results/metrics.json` kaynağından okunur ve `reports/` altına özetlenir.
