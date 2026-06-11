"""
data_utils.py — Data Processing Utilities for Traffic Forecasting

Handles CSV loading, normalization, sequence creation for LSTM input,
synthetic data generation with realistic network traffic patterns, and
congestion probability calculation.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Parsed DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.ParserError: If the CSV is malformed.
    """
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} rows from {filepath}")
    return df


def normalize_data(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Min-max normalize specified columns to the [0, 1] range.

    Args:
        df: Input DataFrame.
        columns: List of column names to normalize.

    Returns:
        Tuple of (normalized DataFrame copy, scaler dict).
        The scaler dict maps column names to {"min": float, "max": float}.
    """
    df_norm = df.copy()
    scaler: Dict[str, Dict[str, float]] = {}

    for col in columns:
        col_min = float(df[col].min())
        col_max = float(df[col].max())

        # Guard against zero-range columns to avoid division by zero
        if col_max - col_min == 0:
            df_norm[col] = 0.0
        else:
            df_norm[col] = (df[col] - col_min) / (col_max - col_min)

        scaler[col] = {"min": col_min, "max": col_max}

    return df_norm, scaler


def denormalize_data(
    values: np.ndarray,
    scaler: Dict[str, Dict[str, float]],
    column: str,
) -> np.ndarray:
    """
    Inverse-transform normalized values back to original scale.

    Args:
        values: Normalized values (numpy array or scalar).
        scaler: Scaler dictionary from normalize_data().
        column: Column name whose scale parameters to use.

    Returns:
        Denormalized values as a numpy array.
    """
    col_min = scaler[column]["min"]
    col_max = scaler[column]["max"]
    return values * (col_max - col_min) + col_min


def create_sequences(
    data: np.ndarray,
    seq_length: int = 24,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sliding-window input/output sequences for LSTM training.

    Given a 2D array of shape (n_samples, n_features), produces:
      - X: (num_sequences, seq_length, n_features)  — input windows
      - y: (num_sequences, n_features)               — next-step targets

    Args:
        data: 2D numpy array of time-series features.
        seq_length: Number of time steps per input window.

    Returns:
        Tuple of (X, y) as float32 PyTorch tensors.
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for i in range(len(data) - seq_length):
        xs.append(data[i : i + seq_length])
        ys.append(data[i + seq_length])

    X = torch.tensor(np.array(xs), dtype=torch.float32)
    y = torch.tensor(np.array(ys), dtype=torch.float32)

    return X, y


def generate_synthetic_data(n_points: int = 500) -> pd.DataFrame:
    """
    Generate realistic synthetic industrial network traffic data.

    Patterns modelled:
      - Diurnal cycle: higher traffic during business hours (08:00–18:00)
      - Random congestion spikes: brief surges in bandwidth and latency
      - Correlated metrics: packet count scales with bandwidth, latency
        increases under high utilization
      - Gaussian noise for natural variability

    Args:
        n_points: Number of data points to generate.

    Returns:
        DataFrame with columns:
          timestamp, bandwidth_utilization, packet_count, latency_ms, is_congested
    """
    np.random.seed(42)

    # Timestamps at 5-minute intervals
    timestamps = pd.date_range(
        start="2025-01-01 00:00:00",
        periods=n_points,
        freq="5min",
    )

    hours = np.array([t.hour + t.minute / 60.0 for t in timestamps])

    # --- Bandwidth Utilization (%) ---
    # Diurnal base: peaks around 13:00, trough at ~04:00
    diurnal = 40 + 30 * np.sin(2 * np.pi * (hours - 6) / 24)

    # Weekly modulation: lower on weekends (period ~7 days = 2016 5-min slots)
    day_index = np.arange(n_points)
    weekly = 5 * np.sin(2 * np.pi * day_index / 2016)

    # Random congestion spikes (~5% of time steps)
    spike_mask = np.random.random(n_points) < 0.05
    spikes = spike_mask * np.random.uniform(15, 35, n_points)

    # Gaussian noise
    noise_bw = np.random.normal(0, 3, n_points)

    bandwidth = np.clip(diurnal + weekly + spikes + noise_bw, 5, 100)

    # --- Packet Count (packets/sec) ---
    # Correlated with bandwidth plus independent noise
    packet_base = bandwidth * np.random.uniform(80, 120, n_points)
    noise_pkt = np.random.normal(0, 200, n_points)
    packet_count = np.clip(packet_base + noise_pkt, 100, 12000).astype(int)

    # --- Latency (ms) ---
    # Baseline latency increases with bandwidth utilization
    latency_base = 5 + 0.3 * bandwidth
    # Additional latency during congestion spikes
    latency_spike = spike_mask * np.random.uniform(10, 50, n_points)
    noise_lat = np.random.normal(0, 2, n_points)
    latency = np.clip(latency_base + latency_spike + noise_lat, 1, 100)

    # --- Congestion Flag ---
    # Mark as congested when bandwidth > 75% OR latency > 40ms
    is_congested = ((bandwidth > 75) | (latency > 40)).astype(int)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "bandwidth_utilization": np.round(bandwidth, 2),
            "packet_count": packet_count,
            "latency_ms": np.round(latency, 2),
            "is_congested": is_congested,
        }
    )

    return df


def calculate_congestion_probability(
    predictions: List[Dict[str, float]],
    thresholds: Optional[Dict[str, float]] = None,
) -> float:
    """
    Estimate the probability of network congestion over a forecast horizon.

    Checks each predicted time step against threshold values; the
    probability is the fraction of steps that exceed any threshold.

    Args:
        predictions: List of dicts, each containing:
                     {"bandwidth_utilization", "packet_count", "latency_ms"}
        thresholds: Optional dict of threshold values.
                    Defaults: bandwidth > 75%, packets > 9000, latency > 40ms.

    Returns:
        Float in [0, 1] representing congestion probability.
    """
    if not predictions:
        return 0.0

    if thresholds is None:
        thresholds = {
            "bandwidth_utilization": 75.0,
            "packet_count": 9000.0,
            "latency_ms": 40.0,
        }

    congested_steps = 0

    for pred in predictions:
        bw = pred.get("bandwidth_utilization", 0)
        pkt = pred.get("packet_count", 0)
        lat = pred.get("latency_ms", 0)

        if (
            bw > thresholds["bandwidth_utilization"]
            or pkt > thresholds["packet_count"]
            or lat > thresholds["latency_ms"]
        ):
            congested_steps += 1

    return round(congested_steps / len(predictions), 4)
