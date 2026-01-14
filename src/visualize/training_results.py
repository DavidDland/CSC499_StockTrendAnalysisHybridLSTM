"""
training_results.py — compare LSTM vs Hybrid training performance from CSV logs
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# -------------------------------
# CONFIGURATION
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

# Fetch latest LSTM and Hybrid CSV logs
def get_latest_csv(model_type):
    csv_list = list(RESULTS_DIR.glob(f"{model_type}_training_log*.csv"))
    if not csv_list:
        print(f"[!] No logs found for {model_type.upper()} model. Run training first.")
        return None
    return max(csv_list, key=lambda f: f.stat().st_mtime)

lstm_csv = get_latest_csv("lstm")
hybrid_csv = get_latest_csv("hybrid")

if not lstm_csv and not hybrid_csv:
    raise FileNotFoundError("No training logs found for either model in /results/")

# Load dataframes
if lstm_csv:
    lstm_df = pd.read_csv(lstm_csv)
    print(f"Loaded LSTM log: {lstm_csv.name}")
if hybrid_csv:
    hybrid_df = pd.read_csv(hybrid_csv)
    print(f"Loaded Hybrid log: {hybrid_csv.name}")

# Helper to set integer ticks
def set_integer_ticks(ax, df_list):
    max_epoch = max(df["epoch"].max() for df in df_list if df is not None)
    ax.set_xticks(np.arange(1, max_epoch + 1, 1))  # one tick per epoch
    ax.set_xlim(1, max_epoch)

# -------------------------------
# PLOT LOSS
# -------------------------------
plt.figure(figsize=(10, 6))
if lstm_csv:
    plt.plot(lstm_df["epoch"], lstm_df["val_loss"], label="LSTM - Val Loss", linestyle="--", linewidth=2)
    plt.plot(lstm_df["epoch"], lstm_df["train_loss"], label="LSTM - Train Loss", linewidth=1.8)
if hybrid_csv:
    plt.plot(hybrid_df["epoch"], hybrid_df["val_loss"], label="Hybrid - Val Loss", linestyle="--", linewidth=2)
    plt.plot(hybrid_df["epoch"], hybrid_df["train_loss"], label="Hybrid - Train Loss", linewidth=1.8)
plt.title("Training vs Validation Loss (LSTM vs Hybrid)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
set_integer_ticks(plt.gca(), [lstm_df if lstm_csv else None, hybrid_df if hybrid_csv else None])
plt.tight_layout()
plt.show()

# -------------------------------
# PLOT ACCURACY
# -------------------------------
plt.figure(figsize=(10, 6))
if lstm_csv:
    plt.plot(lstm_df["epoch"], lstm_df["val_acc"], label="LSTM - Val Acc", linestyle="--", linewidth=2)
    plt.plot(lstm_df["epoch"], lstm_df["train_acc"], label="LSTM - Train Acc", linewidth=1.8)
if hybrid_csv:
    plt.plot(hybrid_df["epoch"], hybrid_df["val_acc"], label="Hybrid - Val Acc", linestyle="--", linewidth=2)
    plt.plot(hybrid_df["epoch"], hybrid_df["train_acc"], label="Hybrid - Train Acc", linewidth=1.8)
plt.title("Training vs Validation Accuracy (LSTM vs Hybrid)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
set_integer_ticks(plt.gca(), [lstm_df if lstm_csv else None, hybrid_df if hybrid_csv else None])
plt.tight_layout()
plt.show()
