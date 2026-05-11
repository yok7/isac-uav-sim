"""
Utility functions for HYPERDOA.

This module provides common utilities for reproducibility and device management.
"""

import random
import numpy as np
import torch


# Constants for angle conversion
R2D = 180.0 / np.pi  # Radians to degrees
D2R = np.pi / 180.0  # Degrees to radians


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries.

    Sets seeds for Python's random module, NumPy, and PyTorch (including CUDA).

    Args:
        seed: Random seed value

    Example:
        >>> set_seed(42)
        >>> np.random.rand()  # Will always produce the same value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = None) -> torch.device:
    """Get compute device for PyTorch operations.

    Args:
        device: Device string ("cuda", "cpu", or None for auto-detect)

    Returns:
        torch.device object

    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("cpu")  # Force CPU
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
