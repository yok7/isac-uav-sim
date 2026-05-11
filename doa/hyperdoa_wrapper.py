"""HyperDOA integration wrapper for ISAC simulation framework.

Provides a lightweight interface to HyperDOA (HDC-based DOA estimation) for
integration with the Sionna RT + OFDM end-to-end ISAC simulation.

Key differences from classical MUSIC:
- HyperDOA is a learning-based method (no eigendecomposition needed)
- Works with snapshot matrices [N, T] directly
- Trained on-the-fly from a training set (one-shot centroid learning)
- Requires torch-hd library for hyperdimensional encoding

Reference: HYPERDOA (arXiv), HyperDOA GitHub repo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

# Import HyperDOA from third-party
HYPERDOA_PATH = Path(__file__).parent / "HYPERDOA"
if str(HYPERDOA_PATH) not in sys.path:
    sys.path.insert(0, str(HYPERDOA_PATH))

try:
    from hyperdoa import (
        HDCAoAModel,
        DOAConfig,
        set_seed,
        D2R,
        R2D,
    )
except ImportError as e:
    raise ImportError(
        f"HyperDOA not found. Install dependencies:\n"
        f"  pip install torch-hd\n"
        f"  cd {HYPERDOA_PATH} && pip install -e .\n"
        f"Original error: {e}"
    )


@dataclass
class HyperDOAConfig:
    """Configuration for HyperDOA integration.

    Attributes:
        num_antennas: Number of ULA sensor elements (N)
        num_sources: Number of signal sources (M), default 1 for ISAC
        num_snapshots: Number of time snapshots (T)
        feature_type: "lag" (default) for non-coherent, "spatial_smoothing" for coherent
        n_dimensions: HDC hypervector dimensionality (default 10000)
        min_angle_deg: Lower bound of DOA search space (default -90)
        max_angle_deg: Upper bound of DOA search space (default 90)
        precision_deg: DOA bin resolution in degrees (default 0.1)
        min_separation_deg: Minimum separation for multi-source peak decoding
        device: "cuda" or "cpu"
        snr_db: SNR for training data generation
        training_samples: Number of training samples for one-shot learning
    """

    num_antennas: int = 8
    num_sources: int = 1
    num_snapshots: int = 100
    feature_type: Literal["lag", "spatial_smoothing"] = "lag"
    n_dimensions: int = 10000
    min_angle_deg: float = -90.0
    max_angle_deg: float = 90.0
    precision_deg: float = 0.1
    min_separation_deg: float = 6.0
    device: str | None = None
    snr_db: float = 10.0
    training_samples: int = 1000

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class HyperDOAEstimator:
    """HyperDOA-based DOA estimator for ISAC.

    Wraps the HyperDOA HDCAoAModel and provides a numpy-friendly interface.
    Model is trained once on synthetic data and cached for reuse.

    Integration with ISAC simulation:
        The ISAC received signal [num_rx_ant, num_symbols, num_subcarriers]
        is compressed into a snapshot matrix [N, T] by averaging over
        subcarriers (or selecting a representative subcarrier), then passed
        to HyperDOA for angle estimation.

    For single-source ISAC scenarios (M=1), set num_sources=1 and use
    predict_single() which returns a scalar angle in degrees.

    Example:
        >>> config = HyperDOAConfig(num_antennas=8, num_sources=1, device="cuda")
        >>> estimator = HyperDOAEstimator(config)
        >>> # Train on synthetic data (one-shot, fast)
        >>> estimator.train()
        >>> # Estimate DOA from ISAC received signal
        >>> y_signal = np.random.randn(8, 14, 128) + 1j*np.random.randn(8, 14, 128)
        >>> angle_deg = estimator.predict_single(y_signal)
        >>> print(f"DOA = {angle_deg:.2f} deg")
    """

    def __init__(
        self,
        config: HyperDOAConfig | None = None,
        seed: int = 42,
        train_data_path: str | Path | None = None,
    ):
        """Initialize HyperDOA estimator.

        Args:
            config: HyperDOA configuration
            seed: Random seed for reproducibility
            train_data_path: Optional path to pre-generated training data (.pt file)
                If None, training data is generated on-the-fly using SubspaceNet
        """
        self.config = config or HyperDOAConfig()
        self.seed = seed
        self._model: torch.nn.Module | None = None
        self._trained = False

        self._device = torch.device(self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Training data path (for reuse across instantiations)
        self._train_data_path = train_data_path

    def _get_train_data(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Load or generate training data.

        Returns:
            List of (X, Y) tuples: X [N, T] complex, Y [M] radians
        """
        if self._train_data_path is not None:
            path = Path(self._train_data_path)
            if path.exists():
                data = torch.load(path, map_location=self._device)
                # Format: list of (X, Y) tuples
                return data

        # Generate on-the-fly using SubspaceNet
        try:
            return self._generate_train_data()
        except ImportError as e:
            raise ImportError(
                f"Cannot generate training data (SubspaceNet not available): {e}\n"
                f"Provide train_data_path or install SubspaceNet."
            ) from e

    def _generate_train_data(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate training data using SubspaceNet signal model.

        Returns:
            List of (X, Y) tuples
        """
        _ssp_dir = Path(__file__).parent / "SubspaceNet"
        sys.path.insert(0, str(_ssp_dir))
        sys.path.insert(0, str(_ssp_dir / "src"))
        from system_model import SystemModelParams
        from signal_creation import Samples

        params = SystemModelParams()
        params.N = self.config.num_antennas
        params.M = self.config.num_sources
        params.T = self.config.num_snapshots
        params.snr = self.config.snr_db
        params.signal_nature = "non-coherent" if self.config.feature_type == "lag" else "coherent"
        params.signal_type = "NarrowBand"
        params.eta = 0.0
        params.bias = 0.0
        params.sv_noise_var = 0.0

        set_seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        samples_model = Samples(params)
        dataset = []

        for _ in range(self.config.training_samples):
            samples_model.set_doa(None)
            X_np, _, _, _ = samples_model.samples_creation(
                noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
            )
            X = torch.tensor(X_np, dtype=torch.complex64).to(self._device)
            Y = torch.tensor(samples_model.doa, dtype=torch.float64)
            dataset.append((X, Y))

        return dataset

    def train(self, train_data=None) -> "HyperDOAEstimator":
        """Train the HDC model using one-shot centroid learning.

        Args:
            train_data: Optional list of (X, Y) tuples. If None, loads/generates data.

        Returns:
            self (for chaining)
        """
        set_seed(self.seed)

        if train_data is None:
            train_data = self._get_train_data()

        self._model = HDCAoAModel(
            N=self.config.num_antennas,
            M=self.config.num_sources,
            T=self.config.num_snapshots,
            feature_type=self.config.feature_type,
            n_dimensions=self.config.n_dimensions,
            min_angle=self.config.min_angle_deg,
            max_angle=self.config.max_angle_deg,
            precision=self.config.precision_deg,
            min_separation_deg=self.config.min_separation_deg,
            device=self._device,
        )

        # One-shot centroid learning: collect all data into batches
        # SubspaceNet returns X=[N,T] and Y=[M] per sample
        # HyperDOA fit() expects X=[batch,N,T], y=class indices [batch]
        X_list, Y_list = [], []
        for X_samp, Y_samp in train_data:
            X_list.append(X_samp)  # [N,T] complex
            Y_list.append(Y_samp)  # [M] float → class index

        X_all = torch.stack(X_list).to(self._device)          # [B, N, T]
        y_all = self._labels_to_indices(Y_list)                 # [B]

        # Power normalize so train/test share same signal level
        power = torch.mean(torch.abs(X_all) ** 2)
        if power > 0:
            X_all = X_all / torch.sqrt(power)

        self._model.fit(X_all, y_all)

        self._trained = True
        return self

    def _labels_to_indices(self, labels: list[torch.Tensor]) -> torch.Tensor:
        """Convert angle labels (degrees) to class indices.

        Args:
            labels: List of [M] tensors (radians), each entry is a batch item.

        Returns:
            class_indices: [B] long tensor
        """
        precision = self.config.precision_deg
        min_angle = self.config.min_angle_deg
        max_angle = self.config.max_angle_deg
        num_classes = int(round((max_angle - min_angle) / precision)) + 1

        indices = []
        for y in labels:
            # y is in radians, convert to degrees
            angle_deg = float(y[0].item() * R2D)
            idx = round((angle_deg - min_angle) / precision)
            idx = max(0, min(idx, num_classes - 1))
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long, device=self._device)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Estimate DOA angles from received signal snapshots.

        Args:
            X: Received signal [num_antennas, num_snapshots] complex (numpy)
               Can also be [num_antennas, num_symbols, num_subcarriers] in which
               case it is averaged over subcarriers to produce [N, T].

        Returns:
            angles_deg: DOA estimates in degrees [num_sources]
        """
        if not self._trained:
            self.train()

        X_tensor = self._prepare_input(X)  # [N, T] complex

        with torch.no_grad():
            try:
                out = self._model.predict(X_tensor.unsqueeze(0))
            except Exception:
                out = self._model.predict(X_tensor)

        if torch.is_tensor(out):
            out_np = out.detach().cpu().numpy()
        else:
            out_np = np.asarray(out)

        angles_deg = out_np.reshape(-1) * R2D
        return angles_deg

    def predict_single(self, X: np.ndarray) -> float:
        """Estimate single DOA angle from received signal.

        Convenience method for ISAC single-target scenarios.

        Args:
            X: Received signal [num_antennas, num_snapshots] or
               [num_antennas, num_symbols, num_subcarriers]

        Returns:
            doa_deg: Single DOA estimate in degrees
        """
        return float(self.predict(X)[0])

    def _prepare_input(self, X: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch complex tensor for HyperDOA.

        Args:
            X: numpy array, can be:
               - [N, T] complex → used directly
               - [N, num_symbols, num_subcarriers] complex → average over subcarriers

        Returns:
            X_torch: torch.complex64 tensor [N, T]
        """
        if X.ndim == 3:
            # [N, num_symbols, num_subcarriers] → average over subcarriers → [N, T]
            X = np.mean(X, axis=2)  # [N, num_symbols]

        if hasattr(X, "detach"):
            X_torch = X.to(self._device)
        else:
            X_torch = torch.tensor(X, dtype=torch.complex64, device=self._device)

        return self._align_snapshots(X_torch)

    def _align_snapshots(self, X: torch.Tensor) -> torch.Tensor:
        """Pad or truncate X to [N, self.config.num_snapshots].

        Handles cases where the actual snapshot count differs from the
        training-time configured value due to mask/filter variation.
        """
        target_T = self.config.num_snapshots
        current_T = X.shape[1]

        if current_T == target_T:
            return X
        if current_T > target_T:
            return X[:, :target_T]
        pad = torch.zeros(
            X.shape[0], target_T - current_T,
            dtype=X.dtype, device=X.device,
        )
        return torch.cat([X, pad], dim=1)

    def save_model(self, path: str | Path) -> None:
        """Save trained model to checkpoint.

        Args:
            path: Destination path (.pt file)
        """
        if not self._trained:
            raise RuntimeError("Model not trained yet")
        torch.save(self._model.state_dict(), path)

    def load_model(self, path: str | Path) -> "HyperDOAEstimator":
        """Load trained model from checkpoint.

        Args:
            path: Source path (.pt file)

        Returns:
            self
        """
        if self._model is None:
            self._model = HDCAoAModel(
                N=self.config.num_antennas,
                M=self.config.num_sources,
                T=self.config.num_snapshots,
                feature_type=self.config.feature_type,
                n_dimensions=self.config.n_dimensions,
                min_angle=self.config.min_angle_deg,
                max_angle=self.config.max_angle_deg,
                precision=self.config.precision_deg,
                min_separation_deg=self.config.min_separation_deg,
                device=self._device,
            )
        self._model.load_state_dict(torch.load(path, map_location=self._device))
        self._trained = True
        return self

    def summary(self) -> str:
        """Return configuration summary."""
        return (
            f"HyperDOAEstimator(config=HyperDOAConfig(\n"
            f"  num_antennas={self.config.num_antennas},\n"
            f"  num_sources={self.config.num_sources},\n"
            f"  num_snapshots={self.config.num_snapshots},\n"
            f"  feature_type='{self.config.feature_type}',\n"
            f"  n_dimensions={self.config.n_dimensions},\n"
            f"  precision_deg={self.config.precision_deg},\n"
            f"  snr_db={self.config.snr_db},\n"
            f"  training_samples={self.config.training_samples}),\n"
            f"  device={self._device},\n"
            f"  trained={self._trained})"
        )


# ---------------------------------------------------------------------------
# Sanity-check utilities (not part of the estimator class)
# ---------------------------------------------------------------------------


def sanity_check_hyperdoa_sign(
    estimator: "HyperDOAEstimator",
    test_angles_deg: tuple[float, ...] = (-30.0, 0.0, 30.0),
    num_ant: int = 8,
    num_snapshots: int = 128,
    snr_db: float = 20.0,
) -> dict[float, float]:
    """Check whether HyperDOA angle sign matches the ULA steering-vector convention.

    Generates clean ULA single-source narrowband snapshots at known angles and
    compares HyperDOA predictions against MUSIC on the same data.  This validates
    the angle-coordinate system without involving Sionna RT, range compensation,
    or multipath.

    Args:
        estimator: Trained HyperDOAEstimator instance.
        test_angles_deg: Angles to test (default: -30, 0, +30 deg).
        num_ant: Number of ULA elements.
        num_snapshots: Number of time snapshots.
        snr_db: Additive noise level (high by default for clean test).

    Returns:
        Dict mapping true_angle → predicted_angle for each test angle.
    """
    from .classical import music_estimate, steering_vector_ula

    wavelength = 1.0
    d = 0.5  # half-wavelength spacing

    results = {}
    noise_std = 10 ** (-snr_db / 20.0)

    for true_deg in test_angles_deg:
        angle_rad = np.deg2rad(true_deg)
        # Steering vector for ULA
        a = np.exp(1j * 2 * np.pi * d * np.arange(num_ant) * np.sin(angle_rad))
        # Random narrowband signal
        s = np.random.randn(num_snapshots) + 1j * np.random.randn(num_snapshots)
        X = a[:, None] * s[None, :] + noise_std * (
            np.random.randn(num_ant, num_snapshots)
            + 1j * np.random.randn(num_ant, num_snapshots)
        )
        X_hyperdoa = X / np.sqrt(np.mean(np.abs(X) ** 2))

        # HyperDOA estimate
        pred_hyperdoa = estimator.predict_single(X_hyperdoa)

        # MUSIC estimate for comparison
        music_ang = float(music_estimate(X @ X.conj().T / num_snapshots, num_sources=1, num_elements=num_ant, angle_grid_deg=np.linspace(-90, 90, 361))[0])

        results[true_deg] = pred_hyperdoa
        print(
            f"  true={true_deg:6.1f} deg  |  MUSIC={music_ang:6.2f} deg  |  HyperDOA={pred_hyperdoa:6.2f} deg"
        )

    # Detect sign flip: if all errors have same sign, HyperDOA might be inverted
    errors = {true: results[true] - true for true in test_angles_deg}
    all_positive = all(e > 0 for e in errors.values())
    all_negative = all(e < 0 for e in errors.values())
    if all_positive or all_negative:
        print(
            "[HyperDOA sanity-check] WARNING: HyperDOA outputs appear to have opposite sign to ULA convention. "
            "Consider applying a sign flip (multiply by -1) in predict_single()."
        )
    else:
        print("[HyperDOA sanity-check] OK: HyperDOA angle sign matches ULA steering-vector convention.")

    return results