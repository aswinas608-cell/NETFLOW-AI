"""
model.py — LSTM Model Definition for Industrial Network Traffic Forecasting

This module defines a PyTorch LSTM-based neural network for predicting
network traffic metrics (bandwidth utilization, packet count, latency).
Includes training, inference, and model persistence utilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple


class TrafficLSTM(nn.Module):
    """
    LSTM-based model for multi-variate time-series traffic forecasting.

    Predicts future values of 3 network metrics:
      - bandwidth_utilization (%)
      - packet_count (packets/sec)
      - latency_ms (milliseconds)

    Architecture:
      Input → LSTM (2 layers, hidden=64) → Dropout → Fully Connected → Output
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 3,
    ) -> None:
        super(TrafficLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Multi-layer LSTM with dropout between layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Dropout before the fully connected layer for regularization
        self.dropout = nn.Dropout(p=dropout)

        # Fully connected output layer maps hidden state to predictions
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM network.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size) — predicted
            values for the next time step.
        """
        # LSTM forward: output shape (batch, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last time step's output for prediction
        last_hidden = lstm_out[:, -1, :]

        # Apply dropout and fully connected layer
        out = self.dropout(last_hidden)
        out = self.fc(out)

        return out


def predict_future(
    model: TrafficLSTM,
    input_sequence: torch.Tensor,
    n_steps: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Autoregressively predict `n_steps` future time steps.

    The model predicts one step at a time, appending each prediction
    to the input window and sliding forward.

    Args:
        model: Trained TrafficLSTM model.
        input_sequence: Tensor of shape (seq_length, input_size) — the
                        seed sequence used to kick off prediction.
        n_steps: Number of future time steps to forecast.
        device: Torch device (cpu/cuda). Defaults to model's device.

    Returns:
        Tensor of shape (n_steps, output_size) containing predictions.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    predictions: List[torch.Tensor] = []

    # Clone to avoid mutating the original input
    current_seq = input_sequence.clone().to(device)

    with torch.no_grad():
        for _ in range(n_steps):
            # Add batch dimension: (1, seq_length, input_size)
            x = current_seq.unsqueeze(0)

            # Get single-step prediction: (1, output_size)
            pred = model(x)
            predictions.append(pred.squeeze(0))

            # Slide window: drop oldest step, append prediction
            current_seq = torch.cat(
                [current_seq[1:], pred.squeeze(0).unsqueeze(0)], dim=0
            )

    return torch.stack(predictions)


def train_model(
    model: TrafficLSTM,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> List[float]:
    """
    Train the LSTM model on the provided sequence data.

    Args:
        model: TrafficLSTM model instance.
        train_data: Tuple of (X, y) tensors where:
                    X shape = (num_samples, seq_length, input_size)
                    y shape = (num_samples, output_size)
        epochs: Number of training epochs.
        lr: Learning rate for the Adam optimizer.
        batch_size: Mini-batch size for training.
        device: Torch device. Defaults to CPU.

    Returns:
        List of average epoch losses (MSE).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    X_train, y_train = train_data
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # Build a DataLoader for mini-batch training
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses: List[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()

            # Gradient clipping to prevent exploding gradients in LSTMs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)

        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch + 1}/{epochs}] — Loss: {avg_loss:.6f}")

    # Switch back to eval mode after training
    model.eval()
    return losses


def save_model(model: TrafficLSTM, path: str) -> None:
    """
    Save the model's state dictionary and architecture metadata.

    Args:
        model: Trained TrafficLSTM model.
        path: File path for the .pth checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "output_size": model.output_size,
    }
    torch.save(checkpoint, path)
    print(f"[INFO] Model saved to {path}")


def load_model(
    path: str,
    device: Optional[torch.device] = None,
) -> TrafficLSTM:
    """
    Load a TrafficLSTM model from a checkpoint file.

    Automatically reconstructs the model architecture from saved
    metadata before loading weights.

    Args:
        path: Path to the .pth checkpoint file.
        device: Target device. Defaults to CPU.

    Returns:
        Loaded TrafficLSTM model in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model = TrafficLSTM(
        input_size=checkpoint.get("input_size", 3),
        hidden_size=checkpoint.get("hidden_size", 64),
        num_layers=checkpoint.get("num_layers", 2),
        output_size=checkpoint.get("output_size", 3),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[INFO] Model loaded from {path}")
    return model
