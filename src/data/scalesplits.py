import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


# CONFIGURATION
DATA_PATH = Path("../../data/processed/merged_prices_with_sentiment_ready.csv")
OUTPUT_DIR = Path("../../data/processed")
CONFIG_DIR = Path("../../configs")

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1


def scale_and_save(df, feature_cols, suffix):
    """Helper function to scale and save a dataset variant."""
    n = len(df)
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        scaled = split_df.copy()
        scaled[feature_cols] = scaler.transform(split_df[feature_cols])
        scaled.to_csv(OUTPUT_DIR / f"{name}_{suffix}.csv", index=False)

    # Save scaler for inference consistency
    with open(OUTPUT_DIR / f"scaler_{suffix}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"[+] Scaled and saved {suffix} splits and scaler.")


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    # -----------------------------
    # Feature group definitions
    # -----------------------------
    market_features = ["daily_return_%", "range_%", "vol_z"]
    sentiment_features = ["positive", "negative", "neutral"]
    hybrid_features = market_features + sentiment_features

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_DIR / "features_lstm.txt", "w") as f:
        f.write("\n".join(market_features))
    with open(CONFIG_DIR / "features_hybrid.txt", "w") as f:
        f.write("\n".join(hybrid_features))
    print("[+] Saved feature config files for LSTM and Hybrid models.")

    # -----------------------------
    # Scale and save both versions
    # -----------------------------
    scale_and_save(df, market_features, "lstm")
    scale_and_save(df, hybrid_features, "hybrid")

    print(" All dataset variants processed successfully.")


if __name__ == "__main__":
    main()
