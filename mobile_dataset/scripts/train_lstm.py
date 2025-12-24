"""
Sequence-to-one LSTM forecaster that leverages engineered features.

Usage examples:
    python train_lstm.py --cell-id 1129
    python train_lstm.py --max-search-cells 30        # find best cell among first 30 IDs
    python train_lstm.py --search-cells 1129,2552,5339
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH = RESULTS_PATH / "lstm_model.pt"
DEFAULT_FEATURE_FILE = ROOT / "data" / "processed" / "milano_features.csv"
DEFAULT_FEATURES = [
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_12",
    "rolling_mean_24",
    "rolling_std_3",
    "rolling_std_6",
    "rolling_std_12",
    "rolling_std_24",
    "diff_1",
    "diff_24",
    "pct_change_1",
    "pct_change_24",
]


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        return self.fc(last_step).squeeze(-1)


def build_sequences(features: np.ndarray, target: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx in range(window, len(target)):
        X.append(features[idx - window : idx])
        y.append(target[idx])
    return np.array(X), np.array(y)


def resolve_feature_cols(args: argparse.Namespace, columns: List[str]) -> List[str]:
    if args.feature_cols:
        requested = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    else:
        requested = DEFAULT_FEATURES

    # backward-compat for weekday naming
    cols = set(columns)
    if "weekday" in cols and "day_of_week" not in cols:
        columns = columns + ["day_of_week"]
    if "weekday_sin" in cols and "dow_sin" not in cols:
        columns = columns + ["dow_sin"]
    if "weekday_cos" in cols and "dow_cos" not in cols:
        columns = columns + ["dow_cos"]

    missing = [c for c in requested if c not in columns]
    if missing:
        raise SystemExit(f"İstenen özellikler veri setinde yok: {missing}")
    return requested


def update_metrics(cell_id: int, entry: Dict[str, float]) -> None:
    metrics_path = RESULTS_PATH / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics[f"lstm_cell_{cell_id}"] = entry
    metrics_path.write_text(json.dumps(metrics, indent=2))


def main(args: argparse.Namespace) -> None:
    if torch is None:
        raise SystemExit("PyTorch bulunamadı. `pip install torch` ile kurduktan sonra tekrar deneyin.")

    features_path = Path(args.features_file or DEFAULT_FEATURE_FILE)
    if not features_path.exists():
        raise SystemExit(f"Özellik dosyası bulunamadı: {features_path}\nÖnce feature_engineering.py çalıştırın.")

    df = pd.read_csv(features_path, parse_dates=["time_interval"])
    if "internet_traffic" not in df.columns:
        raise SystemExit("Beklenen 'internet_traffic' kolonu bulunamadı.")

    feature_cols = resolve_feature_cols(args, df.columns.tolist())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_cell(cell_id: int, *, save_model: bool, plot_flag: bool, verbose: bool) -> Optional[Dict[str, float]]:
        series = (
            df.loc[df["square_id"] == cell_id, ["time_interval", "internet_traffic"] + feature_cols]
            .dropna(subset=feature_cols + ["internet_traffic"])
            .sort_values("time_interval")
            .reset_index(drop=True)
        )
        if len(series) <= args.window + 1:
            if verbose:
                print(f"[SKIP] Cell {cell_id}: yeterli veri yok.")
            return None

        train_split = int(len(series) * args.train_ratio)
        if train_split <= args.window or train_split >= len(series):
            if verbose:
                print(f"[SKIP] Cell {cell_id}: train/test bölünemedi (train_ratio ayarını kontrol edin).")
            return None

        feature_values = series[feature_cols].to_numpy(dtype=float)
        target_values = series["internet_traffic"].to_numpy(dtype=float)

        feature_scaler = StandardScaler()
        feature_scaler.fit(feature_values[:train_split])
        feature_scaled = feature_scaler.transform(feature_values)

        target_scaler = StandardScaler()
        target_scaler.fit(target_values[:train_split].reshape(-1, 1))
        target_scaled = target_scaler.transform(target_values.reshape(-1, 1)).flatten()

        seq_features = np.concatenate([target_scaled.reshape(-1, 1), feature_scaled], axis=1)
        X, y = build_sequences(seq_features, target_scaled, args.window)
        if len(X) == 0:
            if verbose:
                print(f"[SKIP] Cell {cell_id}: pencere sonrası hiç örnek kalmadı.")
            return None

        target_times = series["time_interval"].to_numpy()[args.window :]
        split = int(len(X) * args.train_ratio)
        if split == 0 or split >= len(X):
            if verbose:
                print(f"[SKIP] Cell {cell_id}: train/test ayrımı başarısız (window/train_ratio ayarı).")
            return None

        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        times_test = target_times[split:]

        train_loader = DataLoader(
            SequenceDataset(X_train, y_train),
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_dataset = SequenceDataset(X_test, y_test)

        model = LSTMRegressor(
            input_size=X_train.shape[-1],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        model.train()
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(train_loader.dataset)
            if verbose and (epoch + 1) % max(1, args.epochs // 5) == 0:
                print(f"Cell {cell_id} | Epoch {epoch+1}/{args.epochs} - train MSE: {epoch_loss:.4f}")

        model.eval()
        with torch.no_grad():
            preds_scaled = model(test_dataset.x.to(device)).cpu().numpy()

        preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        y_true = target_scaler.inverse_transform(test_dataset.y.numpy().reshape(-1, 1)).flatten()

        entry = {
            "mae": float(mean_absolute_error(y_true, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
            "r2": float(r2_score(y_true, preds)),
            "window": args.window,
            "epochs": args.epochs,
            "features": feature_cols,
        }
        update_metrics(cell_id, entry)

        if save_model:
            torch.save(model.state_dict(), MODEL_PATH)
        if plot_flag and (not args.no_plot):
            plt.figure(figsize=(10, 4))
            plt.plot(times_test, y_true, label="Actual", linewidth=2)
            plt.plot(times_test, preds, label="LSTM", linewidth=1.5)
            plt.title(f"Cell {cell_id} LSTM Forecast vs Actual")
            plt.xlabel("Time")
            plt.ylabel("Internet Traffic")
            plt.legend()
            plt.tight_layout()
            plot_path = RESULTS_PATH / f"lstm_preds_cell_{cell_id}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"📈 Grafik kaydedildi: {plot_path}")

        if verbose:
            print(f"✅ Cell {cell_id} sonuçları: MAE={entry['mae']:.2f}, RMSE={entry['rmse']:.2f}, R²={entry['r2']:.3f}")

        entry_with_cell = entry.copy()
        entry_with_cell["cell_id"] = cell_id
        return entry_with_cell

    if args.cell_id is not None:
        result = train_cell(args.cell_id, save_model=True, plot_flag=True, verbose=True)
        if result is None:
            raise SystemExit("Seçilen hücre için model kurulamadı.")
        print(f"Model kaydedildi: {MODEL_PATH}")
        return

    if args.search_cells:
        candidate_ids = [int(c.strip()) for c in args.search_cells.split(",") if c.strip()]
    else:
        candidate_ids = sorted(df["square_id"].unique())
        if args.search_random:
            rng = np.random.default_rng(args.search_seed)
            rng.shuffle(candidate_ids)
        if args.max_search_cells > 0:
            candidate_ids = candidate_ids[: args.max_search_cells]

    if not candidate_ids:
        raise SystemExit("Değerlendirilecek hücre listesi boş.")

    print(f"[INFO] Otomatik tarama {len(candidate_ids)} hücre üzerinde çalışacak.")
    best: Optional[Dict[str, float]] = None
    for cid in candidate_ids:
        res = train_cell(cid, save_model=False, plot_flag=False, verbose=not args.quiet_search)
        if res is None:
            continue
        if best is None or res["r2"] > best["r2"]:
            best = res

    if best is None:
        raise SystemExit("Hiçbir hücre üzerinde eğitim başarıyla tamamlanmadı.")

    print(f"[INFO] En iyi hücre: {best['cell_id']} (R²={best['r2']:.3f}, RMSE={best['rmse']:.2f})")
    final_res = train_cell(int(best["cell_id"]), save_model=True, plot_flag=True, verbose=True)
    if final_res is None:
        raise SystemExit("En iyi hücre yeniden eğitilemedi (beklenmeyen durum).")
    print(f"Model kaydedildi: {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hourly LSTM forecaster (single cell or auto best).")
    parser.add_argument("--cell-id", type=int, help="Belirli hücre ID'si. Boş bırakılırsa otomatik arama yapılır.")
    parser.add_argument("--window", type=int, default=24, help="Saatlik pencere uzunluğu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--features-file",
        type=str,
        default=str(DEFAULT_FEATURE_FILE),
        help="Özellikli veri yolu (varsayılan: milano_features.csv)",
    )
    parser.add_argument(
        "--feature-cols",
        type=str,
        help="Kullanılacak kolon listesi (virgülle ayrılmış). Boş bırakılırsa varsayılan liste kullanılır.",
    )
    parser.add_argument(
        "--max-search-cells",
        type=int,
        default=25,
        help="Otomatik aramada değerlendirilecek azami hücre sayısı (0=hepsi).",
    )
    parser.add_argument(
        "--search-cells",
        type=str,
        help="Belirli hücre ID'lerini virgülle gir (ör. '1129,2552').",
    )
    parser.add_argument(
        "--search-random",
        action="store_true",
        help="Otomatik aramada hücre listesini rastgele karıştır.",
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=42,
        help="Rastgele karıştırma tohumu.",
    )
    parser.add_argument(
        "--quiet-search",
        action="store_true",
        help="Otomatik aramada her hücre için ayrıntılı log basma.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Grafik üretme.",
    )
    main(parser.parse_args())
