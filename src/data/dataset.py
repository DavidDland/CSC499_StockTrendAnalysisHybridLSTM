import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# CONFIGURATION
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
BATCH_SIZE = 32
SHUFFLE_TRAIN = True
NUM_WORKERS = 0  # set >0 if on Linux


# -----------------------------
# Custom dataset
# -----------------------------
class StockSequenceDataset(Dataset):
    """Loads pre-generated sequence data (.npz)."""
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# Dataloader creation
# -----------------------------
def get_dataloaders(model_type="lstm", batch_size=BATCH_SIZE):
    """
    Returns train, val, test dataloaders for the chosen model type.

    Args:
        model_type (str): 'lstm' or 'hybrid'
        batch_size (int): mini-batch size
    """
    datasets = {
        "train": StockSequenceDataset(DATA_DIR / f"train_seq_{model_type}.npz"),
        "val": StockSequenceDataset(DATA_DIR / f"val_seq_{model_type}.npz"),
        "test": StockSequenceDataset(DATA_DIR / f"test_seq_{model_type}.npz"),
    }

    dataloaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train" and SHUFFLE_TRAIN),
            num_workers=NUM_WORKERS,
            drop_last=False
        )
        for split, ds in datasets.items()
    }

    return dataloaders


# -----------------------------
# Quick sanity check
# -----------------------------
if __name__ == "__main__":
    for model in ["lstm", "hybrid"]:
        print(f"\n[Testing {model.upper()} dataloaders]")
        loaders = get_dataloaders(model_type=model)
        X_batch, y_batch = next(iter(loaders["train"]))
        print(f"{model.upper()} → X: {X_batch.shape}, y: {y_batch.shape}")
