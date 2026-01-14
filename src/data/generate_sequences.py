import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # from src/data → project root
PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIGS = PROJECT_ROOT / "configs"
SEQ_LEN = 28  # tune if needed


def make_sequences(df, feature_cols, seq_len):
    """Create rolling window sequences for LSTM input."""
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["target"].to_numpy(dtype=np.float32)
    Xs, ys = [], []
    for i in range(len(df) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32).reshape(-1, 1)


def run(model_type):
    # pick correct configs and files
    feature_path = CONFIGS / f"features_{model_type}.txt"
    features = [f.strip() for f in open(feature_path).read().splitlines()]

    print(f"[+] Using features from {feature_path}")
    for split in ["train", "val", "test"]:
        csv_path = PROCESSED / f"{split}_{model_type}.csv"
        df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
        X, y = make_sequences(df, features, SEQ_LEN)
        npz_path = PROCESSED / f"{split}_seq_{model_type}.npz"
        np.savez_compressed(npz_path, X=X, y=y)
        print(f"{split}_{model_type}: X {X.shape}, y {y.shape} → saved to {npz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sequence data for LSTM or Hybrid model")
    parser.add_argument("--model", choices=["lstm", "hybrid"], required=True, help="Model type to process")
    args = parser.parse_args()

    run(args.model)
