import torch
import torch.nn as nn
from models.lstm_classifier import LSTMClassifier
from models.hybrid_classifier import HybridClassifier


def count_parameters(model: nn.Module) -> int:
    """Returns the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    B, SEQ_LEN, FEAT, SENT_DIM = 8, 50, 8, 3
    x = torch.randn(B, SEQ_LEN, FEAT)
    s = torch.randn(B, SENT_DIM)

    lstm = LSTMClassifier(FEAT, bidirectional=True)
    print("LSTM output:", lstm(x).shape)
    print("LSTM params:", count_parameters(lstm))

    hybrid = HybridClassifier(FEAT, SENT_DIM, bidirectional=True)
    print("Hybrid output:", hybrid(x, s).shape)
    print("Hybrid params:", count_parameters(hybrid))
