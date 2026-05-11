"""
HYPERDOA: Hyperdimensional Computing for Direction-of-Arrival Estimation

A lightweight, standalone implementation of HDC-based DOA estimation
for uniform linear arrays (ULA).

Key Features:
    - Single-shot training (no iterative backpropagation)
    - Multiple feature extraction strategies
    - Multi-source DOA estimation
    - Compatible with standard DOA datasets

Example:
    >>> from hyperdoa import HDCAoAModel, DOAConfig, evaluate_hdc
    >>> config = DOAConfig(N=8, M=2, T=100)
    >>> model = HDCAoAModel(N=config.N, M=config.M, T=config.T, feature_type="lag")
    >>> model.train_from_dataloader(train_loader)
    >>> predictions = model.predict(test_data)
"""

from .config import DOAConfig
from .utils import set_seed, get_device, R2D, D2R
from .models import (
    HDCAoAModel,
    HDCFeatureEncoder,
    SpatialSmoothingFeature,
    LagFeature,
)
from .evaluation import (
    evaluate_hdc,
    compute_mspe,
    compute_mspe_db,
    save_checkpoint,
    load_checkpoint,
)

__version__ = "1.0.0"
__author__ = "HYPERDOA Authors"

__all__ = [
    # Config
    "DOAConfig",
    # Utils
    "set_seed",
    "get_device",
    "R2D",
    "D2R",
    # Models
    "HDCAoAModel",
    "HDCFeatureEncoder",
    "SpatialSmoothingFeature",
    "LagFeature",
    # Evaluation
    "evaluate_hdc",
    "compute_mspe",
    "compute_mspe_db",
    "save_checkpoint",
    "load_checkpoint",
]
