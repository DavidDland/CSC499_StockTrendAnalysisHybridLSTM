"""
train.py — Generic training loop for LSTM and Hybrid models with Early Stopping,
Weight Decay, Gradient Clipping, Learning Rate Scheduling, and CSV Logging.
"""

import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from src.models import LSTMClassifier, HybridClassifier
from src.data.dataset import get_dataloaders


# -------------------------------
# CONFIGURATION
# -------------------------------
CONFIG = {
    "model_type": "hybrid",        # "lstm" or "hybrid"
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.4,
    "bidirectional": True,
    "lr": 3e-4,
    "epochs": 50,
    "batch_size": 32,
    "patience": 10,
    "min_delta": 1e-4,
    "weight_decay": 1e-4,          # L2 regularization
    "clip_grad_norm": 1.0,         # gradient clipping max norm
    "scheduler_step": 5,           # reduce LR every N epochs
    "scheduler_gamma": 0.7,        # multiply LR by this factor
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": Path(__file__).resolve().parents[2] / "models" / "checkpoints",
    "log_dir": Path(__file__).resolve().parents[2] / "results",
}


# -------------------------------
# TRAIN FUNCTION
# -------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, clip_grad_norm=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        # Handle hybrid input
        if "Hybrid" in model.__class__.__name__:
            x_seq = X_batch[..., :-3]       # market features
            s_vec = X_batch[:, -1, -3:]     # sentiment features (last timestep)
            outputs = model(x_seq, s_vec)
        else:
            outputs = model(X_batch)

        loss = criterion(outputs, y_batch)
        loss.backward()

        # Apply gradient clipping if configured
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if "Hybrid" in model.__class__.__name__:
                x_seq = X_batch[..., :-3]
                s_vec = X_batch[:, -1, -3:]
                outputs = model(x_seq, s_vec)
            else:
                outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return total_loss / total, correct / total


# -------------------------------
# MAIN TRAINING LOOP WITH EARLY STOPPING + CSV LOGGING
# -------------------------------
def main():
    cfg = CONFIG
    device = torch.device(cfg["device"])
    cfg["save_dir"].mkdir(parents=True, exist_ok=True)
    cfg["log_dir"].mkdir(parents=True, exist_ok=True)

    model_type = cfg["model_type"].lower()
    print(f"Using device: {device}")
    print(f"Training model type: {model_type.upper()}")

    loaders = get_dataloaders(model_type=model_type, batch_size=cfg["batch_size"])

    # Initialize model
    if model_type == "lstm":
        model = LSTMClassifier(
            input_size=3,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            bidirectional=cfg["bidirectional"],
        )
    else:
        model = HybridClassifier(
            input_size=3,
            sent_size=3,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            bidirectional=cfg["bidirectional"],
        )

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = StepLR(optimizer, step_size=cfg["scheduler_step"], gamma=cfg["scheduler_gamma"])

    # Early stopping setup
    best_val_loss = float("inf")
    patience_counter = 0
    best_path = cfg["save_dir"] / f"{model_type}_best.pt"

    # CSV logging setup
    log_path = cfg["log_dir"] / f"{model_type}_training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device, cfg["clip_grad_norm"])
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)

        lr = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch}/{cfg['epochs']}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f} | "
              f"LR: {lr:.6f}")

        # Write to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr])

        # Learning rate scheduler
        scheduler.step()

        # Early stopping check
        if best_val_loss - val_loss > cfg["min_delta"]:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f" Improved! Saved best model to {best_path}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{cfg['patience']}")

        if patience_counter >= cfg["patience"]:
            print(f" Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete! Log saved to {log_path}")


if __name__ == "__main__":
    main()
