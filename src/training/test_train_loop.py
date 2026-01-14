"""
test_train_loop.py — sanity check for training setup
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.models import LSTMClassifier
from src.data.dataset import get_dataloaders

def test_one_epoch():
    loaders = get_dataloaders(batch_size=8)
    model = LSTMClassifier(input_size=6, hidden_size=32, num_layers=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    X_batch, y_batch = next(iter(loaders["train"]))
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()

    print(" One training step successful!")
    print(f"Batch output shape: {outputs.shape}")
    print(f"Loss: {loss.item():.6f}")

if __name__ == "__main__":
    test_one_epoch()
