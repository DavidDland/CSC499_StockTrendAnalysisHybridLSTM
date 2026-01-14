"""
HybridClassifier: Combines LSTM sequence encoding with an auxiliary sentiment vector.
Returns logits (use BCEWithLogitsLoss during training).
"""

import torch
import torch.nn as nn
from .lstm_classifier import _init_weights
from typing import Optional

class HybridClassifier(nn.Module):
    """
    A hybrid model that combines LSTM-based sequence encoding with an auxiliary sentiment vector.
    This model is designed for binary classification tasks, where the input consists of time-series data
    and additional sentiment features.
    """

    def __init__(
        self,
        input_size: int,  # Number of features in the input sequence
        sent_size: int,  # Size of the auxiliary sentiment vector
        hidden_size: int = 64,  # Number of hidden units in the LSTM
        num_layers: int = 2,  # Number of LSTM layers
        dropout: float = 0.1,  # Dropout rate for regularization
        bidirectional: bool = False,  # Whether the LSTM is bidirectional
        mlp_hidden: Optional[int] = None,  # Number of hidden units in the MLP head
    ):
        super().__init__()

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch_size, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Calculate the output dimension of the LSTM
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        # Set the default size for the MLP hidden layer if not provided
        if mlp_hidden is None:
            mlp_hidden = max(lstm_out_dim // 2, 32)

        # Define the MLP head for classification
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim + sent_size, mlp_hidden),  # Combine LSTM output and sentiment vector
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(mlp_hidden, 1),  # Output a single logit
        )

        # Initialize weights for the model
        self.apply(_init_weights)
        self.bidirectional = bidirectional

    def forward(self, x_seq: torch.Tensor, s_vec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the HybridClassifier.

        Args:
            x_seq (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len, input_size).
            s_vec (torch.Tensor): Auxiliary sentiment vector of shape (batch_size, sent_size).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, 1).
        """
        # Pass the input sequence through the LSTM
        _, (h_n, _) = self.lstm(x_seq)

        # Extract the final hidden state from the LSTM
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # Concatenate the last hidden states of both directions
        else:
            h = h_n[-1]  # Use the last hidden state of the single direction

        # Concatenate the LSTM output with the sentiment vector
        x = torch.cat([h, s_vec], dim=-1)

        # Pass the combined features through the MLP head
        return self.mlp(x)