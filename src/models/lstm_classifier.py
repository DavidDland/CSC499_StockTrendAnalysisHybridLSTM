"""
LSTMClassifier: A simple PyTorch LSTM-based binary classifier.
Returns raw logits (use BCEWithLogitsLoss during training).
"""

import torch
import torch.nn as nn

def _init_weights(module):
    """
    Initialize weights for the model layers.
    Uses Xavier initialization for Linear layers and sets biases to zero.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class LSTMClassifier(nn.Module):
    """
    A binary classifier based on an LSTM architecture.
    This model processes sequential data and outputs logits for binary classification.
    """

    def __init__(
        self,
        input_size: int,  # Number of features in the input sequence
        hidden_size: int = 64,  # Number of hidden units in the LSTM
        num_layers: int = 2,  # Number of LSTM layers
        dropout: float = 0.1,  # Dropout rate for regularization
        bidirectional: bool = False,  # Whether the LSTM is bidirectional
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch_size, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Define the fully connected head for classification
        head_in = hidden_size * self.num_directions
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, head_in // 2),  # Reduce dimensionality
            nn.ReLU(),
            nn.Linear(head_in // 2, 1),  # Output a single logit
        )

        # Initialize weights for the model
        self.apply(_init_weights)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTMClassifier.

        Args:
            x_seq (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, 1).
        """
        # Pass the input sequence through the LSTM
        _, (h_n, _) = self.lstm(x_seq)

        # Extract the final hidden state
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # Concatenate the last hidden states of both directions
        else:
            h = h_n[-1]  # Use the last hidden state of the single direction

        # Pass the hidden state through the fully connected head
        return self.head(h)