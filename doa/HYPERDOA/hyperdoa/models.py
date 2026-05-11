"""
HDC Models for Direction-of-Arrival Estimation.

This module provides the core HDC (Hyperdimensional Computing) model for
angle-of-arrival estimation using various feature extraction strategies.

Key Components:
    - HDCFeatureEncoder: Base class for feature encoders
    - LagFeature: Mean spatial-lag autocorrelation features
    - SpatialSmoothingFeature: Spatial smoothing covariance features
    - HDCAoAModel: Main unified HDC model for DOA estimation

Requirements:
    - torchhd (pip install torch-hd)
    - torch
    - numpy

Example:
    >>> from hyperdoa import HDCAoAModel
    >>> model = HDCAoAModel(N=8, M=2, T=100, feature_type="lag")
    >>> model.train_from_dataloader(train_loader)
    >>> predictions = model.predict(test_data)
"""

from typing import List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
from itertools import permutations

try:
    import torchhd as hd
except ImportError:
    hd = None


# ============================================================================
# Feature Extraction Modules
# ============================================================================


class HDCFeatureEncoder(nn.Module):
    """Base class for feature encoders with auto-created HDC encoders.

    This abstract class provides the foundation for all feature extractors,
    handling encoder creation and common utilities.

    Args:
        n_features: Number of input features
        n_dimensions: Hypervector dimensionality (default: 10000)
        device: Compute device (default: auto-detect)
    """

    def __init__(
        self,
        n_features: int,
        n_dimensions: int = 10000,
        device: Optional[Union[torch.device, str]] = None,
        **encoder_kwargs,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.encoder_kwargs = encoder_kwargs
        self.encoder, self.encoder_imag = self._create_encoders()

    def _create_encoders(self):
        """Create default HDC encoder(s)."""
        if hd is None:
            raise ImportError(
                "torchhd is required for HDC encoders. Install with: pip install torch-hd"
            )

        encoder = hd.embeddings.FractionalPower(
            in_features=self.n_features,
            out_features=self.n_dimensions,
            distribution="sinc",
            bandwidth=1.0,
            vsa="FHRR",
            device=self.device,
            requires_grad=False,
        )
        return encoder, encoder

    def _zscore(self, x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
        """Z-score normalization."""
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True, correction=0)
        return (x - mean) / (std + eps)

    def _ensure_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is complex type."""
        return x if torch.is_complex(x) else x.to(torch.complex64)

    def extract_features(self, X: torch.Tensor):
        """Extract raw features from input. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement extract_features")

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """Extract features and encode them into hypervectors."""
        raise NotImplementedError("Subclasses must implement encode")


class SpatialSmoothingFeature(HDCFeatureEncoder):
    """Spatial smoothing covariance feature extractor for coherent sources.

    Uses forward-backward spatial smoothing to decorrelate coherent sources
    before computing covariance features.

    Args:
        n_features: Number of features
        n_dimensions: Hypervector dimensionality
        device: Compute device
        center: Whether to center data
        sub_array_ratio: Ratio of sub-array size to full array (0.1 to 1.0)
    """

    def __init__(
        self,
        n_features: int,
        n_dimensions: int = 10000,
        device: Optional[Union[torch.device, str]] = None,
        center: bool = False,
        sub_array_ratio: float = 0.5,
        **encoder_kwargs,
    ):
        self.center = center
        self.sub_array_ratio = max(0.1, min(1.0, sub_array_ratio))
        super().__init__(n_features, n_dimensions, device, **encoder_kwargs)

    def extract_features(self, X: torch.Tensor) -> torch.Tensor:
        """Extract spatial smoothing features.

        Args:
            X: Input tensor of shape (batch, N, T) or (N, T)

        Returns:
            Features tensor
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)
        N, M, T = X.shape
        Xc = X - X.mean(dim=-1, keepdim=True) if self.center else X
        Xc = self._ensure_complex(Xc)
        sub_array_size = int(M * self.sub_array_ratio) + 1
        sub_array_size = max(2, min(sub_array_size, M))
        number_of_sub_arrays = M - sub_array_size + 1
        smoothed_covs = []
        for b in range(N):
            X_sample = Xc[b]
            R_smoothed = torch.zeros(
                (sub_array_size, sub_array_size),
                dtype=X_sample.dtype,
                device=X_sample.device,
            )
            for j in range(number_of_sub_arrays):
                X_sub = X_sample[j : j + sub_array_size, :]
                R_sub = (X_sub @ X_sub.conj().transpose(-2, -1)) / T
                R_smoothed += R_sub
            R_smoothed /= number_of_sub_arrays
            smoothed_covs.append(R_smoothed)
        R = torch.stack(smoothed_covs, dim=0)
        sub_M = R.shape[1]
        rows, cols = torch.triu_indices(sub_M, sub_M, device=R.device)
        tri = R[:, rows, cols]
        feat = torch.cat([tri.real, tri.imag], dim=1)
        return self._zscore(feat.float())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.extract_features(X)

    @torch.no_grad()
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(X)
        return self.encoder(feats)


class LagFeature(HDCFeatureEncoder):
    """Mean spatial-lag autocorrelation features.

    Extracts diagonal elements from the covariance matrix at different lags,
    providing a compact representation of spatial correlation structure.

    Args:
        n_features: Number of features (2 * N for N sensors)
        n_dimensions: Hypervector dimensionality
        device: Compute device
        center: Whether to center data
        normalize_power: Whether to normalize by lag-0 power
    """

    def __init__(
        self,
        n_features: int,
        n_dimensions: int = 10000,
        device: Optional[Union[torch.device, str]] = None,
        center: bool = False,
        normalize_power: bool = False,
        **encoder_kwargs,
    ):
        self.center = center
        self.normalize_power = normalize_power
        super().__init__(n_features, n_dimensions, device, **encoder_kwargs)

    def extract_features(self, X: torch.Tensor) -> torch.Tensor:
        """Extract lag-based features.

        Args:
            X: Input tensor of shape (batch, N, T) or (N, T)

        Returns:
            Features tensor of shape (batch, 2*N)
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)
        Xc = X - X.mean(dim=-1, keepdim=True) if self.center else X
        Xc = self._ensure_complex(Xc)
        N, M, T = Xc.shape
        R = (Xc @ Xc.conj().transpose(-2, -1)) / T
        lags = [
            torch.diagonal(R, offset=k, dim1=-2, dim2=-1).mean(dim=1) for k in range(M)
        ]
        r = torch.stack(lags, dim=1)
        if self.normalize_power:
            denom = r[:, :1].abs().clamp_min(1e-12)
            r = r / denom
        feat = torch.cat([r.real, r.imag], dim=1)
        return self._zscore(feat.float())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.extract_features(X)

    @torch.no_grad()
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(X)
        return self.encoder(feats)


# ============================================================================
# Unified HDC Model
# ============================================================================


class HDCAoAModel(nn.Module):
    """Unified HDC model for Angle-of-Arrival estimation.

    This model combines feature extraction, HDC encoding, and multi-source decoding
    in a single, clean architecture.

    Args:
        N: Number of sensors
        M: Number of sources
        T: Number of time snapshots
        feature_type: Feature extraction method
            - "lag": Mean spatial-lag features
            - "spatial_smoothing": Spatial smoothing covariance
        n_dimensions: Hypervector dimensionality (default: 10000)
        min_angle: Minimum angle in degrees (default: -90)
        max_angle: Maximum angle in degrees (default: 90)
        precision: Angle resolution in degrees (default: 0.1)
        min_separation_deg: Minimum peak separation for multi-source decoding
        tau: Time lag parameter (unused, kept for compatibility)
        device: Compute device

    Example:
        >>> model = HDCAoAModel(N=8, M=2, T=100, feature_type="lag")
        >>> model.train_from_dataloader(train_loader)
        >>> predictions = model.predict(test_data)  # Returns radians
    """

    def __init__(
        self,
        N: int,
        M: int,
        T: int,
        feature_type: str = "lag",
        n_dimensions: int = 10000,
        min_angle: float = -90.0,
        max_angle: float = 90.0,
        precision: float = 0.1,
        min_separation_deg: float = 6.0,
        tau: int = 5,
        device: Optional[Union[torch.device, str]] = None,
        pad_strategy: str = "random",
        share_encoders: bool = False,
    ):
        super().__init__()

        if hd is None:
            raise ImportError(
                "torchhd is required for HDC models. Install with: pip install torch-hd"
            )

        self.N = N
        self.M = M
        self.T = T
        self.n_dimensions = n_dimensions
        self.min_angle = float(min_angle)
        self.max_angle = float(max_angle)
        self.precision = float(precision)
        self.min_separation_deg = float(min_separation_deg)
        self.tau = tau
        self.pad_strategy = pad_strategy
        self.share_encoders = share_encoders
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.n_classes = (
            int(round((self.max_angle - self.min_angle) / self.precision)) + 1
        )

        # Initialize feature encoder based on type
        self.feature_type = feature_type
        if feature_type == "lag":
            self.feature_encoder = LagFeature(
                n_features=N * 2, n_dimensions=n_dimensions, device=self.device
            )
        elif feature_type == "spatial_smoothing":
            sub_array_size = max(2, min(int(N * 0.5) + 1, N))
            self.feature_encoder = SpatialSmoothingFeature(
                n_features=sub_array_size * (sub_array_size + 1),
                n_dimensions=n_dimensions,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unsupported feature_type: {feature_type}. Supported: 'lag', 'spatial_smoothing'"
            )

        self.classifier = hd.models.Centroid(
            in_features=n_dimensions,
            out_features=self.n_classes,
            device=self.device,
        )

        self._reset_centroid_weights()

        # Training hyperparameters
        self.lr = 0.035
        self.epochs = 1
        self.batch_size = 64

        self.to(self.device)

    def _reset_centroid_weights(self):
        """Reset classifier weights to zeros."""
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, self.N, self.T, dtype=torch.complex64, device=self.device
            )
            hv = self.feature_encoder.encode(dummy_input)
            new_w = hv.new_zeros(self.n_classes, self.n_dimensions)
            self.classifier.weight = nn.Parameter(new_w, requires_grad=False)

    def extract_features(self, X: torch.Tensor):
        """Extract features from input data."""
        return self.feature_encoder.extract_features(X)

    @torch.no_grad()
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """Encode input into hypervectors."""
        return self.feature_encoder.encode(X.to(self.device))

    @torch.no_grad()
    def predict_logits(self, X: torch.Tensor) -> torch.Tensor:
        """Get raw classification logits.

        Args:
            X: Input tensor

        Returns:
            Logits tensor of shape (batch, n_classes)
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)
        X = X.to(self.device)
        hv = self.encode(X)
        return self.classifier(hv, dot=True)

    @torch.no_grad()
    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
    ):
        """Train the HDC model.

        Args:
            X_train: Training data tensor
            y_train: Training labels (class indices)
            epochs: Number of training epochs
            lr: Learning rate for centroid updates
            batch_size: Batch size for training

        Returns:
            self for method chaining
        """
        self.train()
        epochs = epochs or self.epochs
        lr = lr or self.lr
        batch_size = batch_size or self.batch_size

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for _ in range(epochs):
            for Xb, yb in loader:
                Xb = Xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                hv = self.encode(Xb)
                if yb.dim() == 1:
                    logits = self.classifier(hv, dot=True)
                    preds = torch.argmax(logits, dim=1)
                    wrong = preds.ne(yb)
                    if wrong.any():
                        pos_idx, neg_idx, vecs = yb[wrong], preds[wrong], hv[wrong]
                        self.classifier.weight.index_add_(0, pos_idx, lr * vecs)
                        self.classifier.weight.index_add_(0, neg_idx, -lr * vecs)
                else:
                    flat_pos_idx, flat_vecs = [], []
                    for i in range(yb.size(0)):
                        for lbl in yb[i].view(-1):
                            li = int(lbl.item())
                            if 0 <= li < self.n_classes:
                                flat_pos_idx.append(li)
                                flat_vecs.append(hv[i : i + 1])
                    if flat_pos_idx:
                        pos_idx = torch.tensor(
                            flat_pos_idx, device=self.device, dtype=torch.long
                        )
                        vecs = torch.cat(flat_vecs, dim=0)
                        self.classifier.weight.index_add_(0, pos_idx, lr * vecs)

        self.classifier.normalize()
        return self

    def _select_peaks_argsort_minsep(
        self, scores: torch.Tensor, k: int, radius: int
    ) -> List[int]:
        """Select peaks using argsort with minimum separation."""
        idx_sorted = torch.argsort(scores, descending=True)
        selected: List[int] = []
        for idx_t in idx_sorted:
            idx = int(idx_t.item())
            ok = all(abs(idx - j) > radius for j in selected)
            if ok:
                selected.append(idx)
                if len(selected) >= k:
                    break
        return selected

    @torch.no_grad()
    def predict_multi(
        self,
        X: torch.Tensor,
        k: Optional[int] = None,
    ) -> List[List[float]]:
        """Predict multiple source angles.

        Args:
            X: Input tensor
            k: Number of sources to detect (default: self.M)

        Returns:
            List of angle lists in degrees
        """
        k = k or self.M
        radius_bins = max(
            1, int(round(self.min_separation_deg / max(self.precision, 1e-12)))
        )

        logits = self.predict_logits(X)
        preds: List[List[float]] = []

        for i in range(logits.shape[0]):
            scores = logits[i]
            idxs = self._select_peaks_argsort_minsep(scores, k, radius_bins)
            angles = [self.min_angle + j * self.precision for j in idxs]
            preds.append(angles)

        return preds

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Predict DOA angles in radians.

        Args:
            X: Input tensor of shape (batch, N, T) or (N, T)

        Returns:
            Predictions in radians, shape (batch, M)
        """
        single = X.dim() == 2
        if single:
            X = X.unsqueeze(0)

        preds_deg = self.predict_multi(X)

        out = []
        for angles in preds_deg:
            a = angles[: self.M]
            while len(a) < self.M:
                if self.pad_strategy == "random":
                    a.append(float(np.round(np.random.rand() * 180.0, 2) - 90.0))
                else:
                    a.append(0.0)
            out.append(a)

        arr = np.array(out, dtype=float)
        return arr * math.pi / 180.0

    @torch.no_grad()
    def compute_mspe_db(self, test_loader: torch.utils.data.DataLoader) -> float:
        """Compute permutation-invariant MSPE in dB on test data.

        Args:
            test_loader: DataLoader with (X, Y) pairs

        Returns:
            Mean MSPE in dB
        """
        total_mspe, count = 0.0, 0
        for Xb, Yb in test_loader:
            preds = self.predict(Xb)
            targets = Yb.detach().cpu().numpy()

            preds = np.asarray(preds)
            targets = np.asarray(targets)
            preds = np.atleast_2d(preds)
            targets = np.atleast_2d(targets)

            for pred_i, target_i in zip(preds, targets):
                M = int(target_i.shape[0])
                best_mspe = float("inf")
                for perm in permutations(pred_i.tolist(), M):
                    p_arr = np.asarray(perm, dtype=float)
                    err = ((p_arr - target_i) + math.pi / 2.0) % math.pi - math.pi / 2.0
                    mspe = (np.linalg.norm(err) ** 2) / M
                    best_mspe = min(best_mspe, mspe)
                total_mspe += best_mspe
                count += 1

        mean_mspe = total_mspe / max(count, 1)
        mean_mspe = max(mean_mspe, 1e-12)  # Avoid log(0)
        return float(10.0 * np.log10(mean_mspe))

    @staticmethod
    def radians_to_indices(
        y_rad: torch.Tensor, min_angle: float, precision: float
    ) -> torch.Tensor:
        """Convert angles in radians to class indices."""
        y_deg = y_rad * 180.0 / math.pi
        idx = torch.round((y_deg - min_angle) / precision).to(torch.long)
        return idx

    def train_from_dataloader(self, train_loader: torch.utils.data.DataLoader) -> None:
        """Train model from a DataLoader.

        Args:
            train_loader: DataLoader yielding (X, Y) batches
        """
        X_list, Y_list = [], []
        for Xb, Yb in train_loader:
            X_list.append(Xb)
            Y_list.append(Yb)

        X = torch.cat(X_list, 0)
        Y = torch.cat(Y_list, 0).squeeze(-1)

        Y_idx = self.radians_to_indices(Y, self.min_angle, self.precision)
        Y_idx = torch.clamp(Y_idx, 0, self.n_classes - 1)

        self.fit(X, Y_idx)
