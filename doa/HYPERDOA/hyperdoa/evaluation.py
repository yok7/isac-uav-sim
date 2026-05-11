"""
Evaluation functions for HDC-based DOA estimation.

This module provides utilities for training and evaluating
HDC models for direction-of-arrival estimation.

Functions:
    compute_mspe: Compute permutation-invariant MSPE
    compute_mspe_db: Compute permutation-invariant MSPE in dB
    evaluate_hdc: Train and evaluate HDC model
"""

import math
import time
from typing import List, Tuple, Optional
from itertools import permutations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models import HDCAoAModel
from .config import DOAConfig
from .utils import set_seed


def compute_mspe(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Permutation-invariant MSPE (Mean Square Periodic Error).

    For multi-source DOA estimation, predictions can be in any order.
    This function finds the best permutation and computes MSPE.

    MSPE = (1/M) * ||error||^2

    Args:
        predictions: Predicted angles in radians, shape (N, M) or (N,)
        targets: Ground truth angles in radians, shape (N, M) or (N,)

    Returns:
        Mean MSPE across all samples

    Example:
        >>> preds = np.array([[0.1, 0.5], [0.2, 0.6]])  # 2 samples, 2 sources
        >>> targets = np.array([[0.5, 0.1], [0.6, 0.2]])
        >>> loss = compute_mspe(preds, targets)  # Order doesn't matter
    """
    predictions = np.atleast_2d(predictions)
    targets = np.atleast_2d(targets)

    total_mspe = 0.0
    count = 0

    for pred_i, target_i in zip(predictions, targets):
        M = len(target_i)
        best_mspe = float("inf")

        for perm in permutations(pred_i.tolist(), M):
            p_arr = np.asarray(perm, dtype=float)
            # Wrap angular difference to [-pi/2, pi/2]
            err = ((p_arr - target_i) + math.pi / 2) % math.pi - math.pi / 2
            mspe = (np.linalg.norm(err) ** 2) / M
            best_mspe = min(best_mspe, mspe)

        total_mspe += best_mspe
        count += 1

    return float(total_mspe / max(count, 1))


def compute_mspe_db(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Permutation-invariant MSPE in dB.

    Converts MSPE to decibel scale: MSPE_dB = 10 * log10(MSPE)

    Args:
        predictions: Predicted angles in radians, shape (N, M) or (N,)
        targets: Ground truth angles in radians, shape (N, M) or (N,)

    Returns:
        Mean MSPE in dB across all samples

    Example:
        >>> preds = np.array([[0.1, 0.5], [0.2, 0.6]])
        >>> targets = np.array([[0.5, 0.1], [0.6, 0.2]])
        >>> loss_db = compute_mspe_db(preds, targets)
    """
    mspe = compute_mspe(predictions, targets)
    # Avoid log(0) by clamping to a small positive value
    mspe = max(mspe, 1e-12)
    return float(10.0 * np.log10(mspe))


def evaluate_hdc(
    train_data: List[Tuple[torch.Tensor, torch.Tensor]],
    test_data: List[Tuple[torch.Tensor, torch.Tensor]],
    config: DOAConfig,
    feature_type: str = "lag",
    device: str = None,
    min_separation_deg: float = 6.0,
    n_dimensions: int = 10000,
    return_model: bool = False,
    verbose: bool = True,
    seed: int = 42,
) -> Tuple[float, Optional[HDCAoAModel]]:
    """Train and evaluate HDC model on given datasets.

    This is the main entry point for HDC-based DOA estimation.
    It handles data loading, model initialization, training, and evaluation.

    Args:
        train_data: Training dataset as list of (X, Y) tuples
            - X: Complex tensor of shape (N, T) - sensor observations
            - Y: Tensor of shape (M,) - ground truth DOA in radians
        test_data: Test dataset in same format as train_data
        config: DOAConfig with system parameters (N, M, T, etc.)
        feature_type: Feature extraction method:
            - "lag": Mean spatial-lag features
            - "spatial_smoothing": Spatial smoothing covariance
        device: Compute device ("cuda", "cpu", or None for auto)
        min_separation_deg: Minimum peak separation in degrees
        n_dimensions: Hypervector dimensionality
        return_model: Whether to return the trained model
        verbose: Print progress messages
        seed: Random seed for reproducibility

    Returns:
        test_loss: MSPE on test set (dB)
        model: Trained HDCAoAModel (if return_model=True, else None)

    Example:
        >>> config = DOAConfig(N=8, M=2, T=100)
        >>> train_data = torch.load("train.pt")
        >>> test_data = torch.load("test.pt")
        >>> loss, model = evaluate_hdc(train_data, test_data, config, return_model=True)
        >>> print(f"Test MSPE: {loss:.2f} dB")
    """
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Initialize HDC model
    hdc = HDCAoAModel(
        N=config.N,
        M=config.M,
        T=config.T,
        feature_type=feature_type,
        n_dimensions=n_dimensions,
        device=torch.device(device),
        min_separation_deg=min_separation_deg,
    )

    # Set training parameters
    hdc.epochs = 1
    hdc.lr = 0.035
    hdc.batch_size = 256 * 16 if device == "cuda" else 64

    # Train
    if verbose:
        print(f"Training HDC ({feature_type})...")
    start_time = time.time()
    hdc.train_from_dataloader(train_loader)
    train_time = time.time() - start_time
    if verbose:
        print(f"  Training time: {train_time:.2f}s")

    # Evaluate - collect predictions and targets
    start_time = time.time()
    all_preds = []
    all_targets = []
    for Xb, Yb in test_loader:
        preds = hdc.predict(Xb)
        all_preds.append(preds)
        all_targets.append(Yb.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute MSPE in dB
    test_mspe_db = compute_mspe_db(all_preds, all_targets)
    eval_time = time.time() - start_time

    if verbose:
        print(f"  Evaluation time: {eval_time:.2f}s")
        print(f"  Test MSPE: {test_mspe_db:.2f} dB")

    if return_model:
        return float(test_mspe_db), hdc
    return float(test_mspe_db), None


def save_checkpoint(model: HDCAoAModel, path: str, meta: dict = None) -> None:
    """Save HDC model checkpoint.

    Args:
        model: Trained HDCAoAModel
        path: Output file path (.pt)
        meta: Optional metadata dictionary
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "meta": {
            "N": model.N,
            "M": model.M,
            "T": model.T,
            "feature_type": model.feature_type,
            "n_dimensions": model.n_dimensions,
            "min_angle": model.min_angle,
            "max_angle": model.max_angle,
            "precision": model.precision,
            "min_separation_deg": model.min_separation_deg,
            **(meta or {}),
        },
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: str = None) -> Tuple[HDCAoAModel, dict]:
    """Load HDC model from checkpoint.

    Args:
        path: Checkpoint file path (.pt)
        device: Target device

    Returns:
        Tuple of (model, metadata)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    meta = checkpoint.get("meta", {})
    state_dict = checkpoint.get("state_dict", checkpoint)

    model = HDCAoAModel(
        N=meta.get("N", 8),
        M=meta.get("M", 2),
        T=meta.get("T", 100),
        feature_type=meta.get("feature_type", "lag"),
        n_dimensions=meta.get("n_dimensions", 10000),
        min_angle=meta.get("min_angle", -90.0),
        max_angle=meta.get("max_angle", 90.0),
        precision=meta.get("precision", 0.1),
        min_separation_deg=meta.get("min_separation_deg", 6.0),
        device=torch.device(device),
    )

    model.load_state_dict(state_dict, strict=False)
    return model, meta
