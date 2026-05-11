"""Joint range and DOA estimation for ISAC systems.

This module provides:
1. JointRangeAngleEstimator: Joint range-DOA estimation
2. RangeAngleSearchGrid: 2D search grid for range-angle space
3. delay_music_spectrum: Delay-domain MUSIC for range estimation
4. coherent_music_spectrum: Coherent MUSIC for multipath suppression

Reference: ISAC range-DOA joint estimation literature
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


C = 299_792_458.0


def steering_vector_ula(
    num_elements: int,
    angle_deg: float,
    d_over_lambda: float = 0.5,
) -> np.ndarray:
    """Compute ULA steering vector."""
    theta = np.deg2rad(angle_deg)
    n = np.arange(num_elements)
    return np.exp(-1j * 2 * np.pi * d_over_lambda * n * np.sin(theta))


@dataclass
class RangeAngleSearchGrid:
    """2D search grid for range-angle estimation.

    Attributes:
        range_min_m: Minimum search range in meters
        range_max_m: Maximum search range in meters
        range_resolution_m: Range resolution in meters
        angle_min_deg: Minimum search angle in degrees
        angle_max_deg: Maximum search angle in degrees
        angle_resolution_deg: Angle resolution in degrees
    """

    range_min_m: float = 0.0
    range_max_m: float = 300.0
    range_resolution_m: float = 0.5
    angle_min_deg: float = -70.0
    angle_max_deg: float = 70.0
    angle_resolution_deg: float = 0.5

    def __post_init__(self):
        self.range_grid = np.arange(self.range_min_m, self.range_max_m + 1e-9, self.range_resolution_m)
        self.angle_grid = np.arange(self.angle_min_deg, self.angle_max_deg + 1e-9, self.angle_resolution_deg)

    @property
    def num_range_bins(self) -> int:
        return len(self.range_grid)

    @property
    def num_angle_bins(self) -> int:
        return len(self.angle_grid)


class JointRangeAngleEstimator:
    """Joint range and angle estimator for ISAC systems.

    Performs 2D matched filtering over range and angle dimensions
    to jointly estimate target range and DOA. Can use either
    conventional beamforming or MUSIC-based approaches.

    Example:
        >>> estimator = JointRangeAngleEstimator(num_elements=8)
        >>> range_est, angle_est, spectrum_2d = estimator.estimate(y, x, valid_mask, f_k)
    """

    def __init__(
        self,
        num_elements: int,
        d_over_lambda: float = 0.5,
        range_grid: RangeAngleSearchGrid | None = None,
    ):
        """Initialize joint estimator.

        Args:
            num_elements: Number of antenna elements
            d_over_lambda: Element spacing
            range_grid: Custom range search grid
        """
        self.num_elements = num_elements
        self.d_over_lambda = d_over_lambda
        self.range_grid = range_grid if range_grid is not None else RangeAngleSearchGrid()

    def estimate_coarse_to_fine(
        self,
        y: np.ndarray,
        x: np.ndarray,
        valid_mask: np.ndarray,
        f_k: np.ndarray,
        num_sources: int = 1,
        coarse_resolution: float = 2.0,
        fine_resolution: float = 0.25,
    ) -> tuple[float, float, np.ndarray]:
        """Coarse-to-fine joint range-angle estimation.

        First performs coarse search over full range-angle space,
        then refines around detected peaks.

        Args:
            y: Received signal [num_rx_ant, num_symbols, num_subcarriers]
            x: Transmitted signal [num_tx_ant, num_symbols, num_subcarriers]
            valid_mask: Valid RE mask [num_symbols, num_subcarriers]
            f_k: Subcarrier frequencies [num_subcarriers]
            num_sources: Number of sources to estimate
            coarse_resolution: Coarse search resolution
            fine_resolution: Fine search resolution

        Returns:
            best_range: Estimated range in meters
            best_angle: Estimated angle in degrees
            delay_profile: 1D delay profile at best angle
        """
        coarse_ranges = np.arange(
            self.range_grid.range_min_m,
            self.range_grid.range_max_m + 1e-9,
            coarse_resolution,
        )
        coarse_angles = np.arange(
            self.range_grid.angle_min_deg,
            self.range_grid.angle_max_deg + 1e-9,
            coarse_resolution,
        )

        best_range = float(coarse_ranges[0])
        best_angle = float(coarse_angles[0])
        best_score = -np.inf
        best_delay_profile = np.zeros_like(coarse_ranges, dtype=float)

        for angle in coarse_angles:
            per_sc = self._per_subcarrier_channel(y, x, valid_mask, float(angle))
            delay_profile = self._delay_profile(per_sc, f_k, coarse_ranges)
            peak_idx = int(np.argmax(delay_profile))
            score = float(delay_profile[peak_idx])

            if score > best_score:
                best_score = score
                best_range = float(coarse_ranges[peak_idx])
                best_angle = float(angle)
                best_delay_profile = delay_profile

        fine_ranges = np.arange(
            max(self.range_grid.range_min_m, best_range - 2.0),
            min(self.range_grid.range_max_m, best_range + 2.0) + 1e-9,
            fine_resolution,
        )
        fine_angles = np.arange(
            max(self.range_grid.angle_min_deg, best_angle - 2.0),
            min(self.range_grid.angle_max_deg, best_angle + 2.0) + 1e-9,
            fine_resolution,
        )

        best_score = -np.inf
        for angle in fine_angles:
            per_sc = self._per_subcarrier_channel(y, x, valid_mask, float(angle))
            delay_profile = self._delay_profile(per_sc, f_k, fine_ranges)
            peak_idx = int(np.argmax(delay_profile))
            score = float(delay_profile[peak_idx])

            if score > best_score:
                best_score = score
                best_range = float(fine_ranges[peak_idx])
                best_angle = float(angle)

        return best_range, best_angle, best_delay_profile

    def _per_subcarrier_channel(
        self,
        y: np.ndarray,
        x: np.ndarray,
        valid_mask: np.ndarray,
        angle_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate per-subcarrier channel for given angle."""
        num_rx_ant = y.shape[0]
        num_tx_ant = x.shape[0]
        num_subcarriers = x.shape[-1]

        a_tx = steering_vector_ula(num_tx_ant, angle_deg, self.d_over_lambda) / np.sqrt(num_tx_ant)
        a_rx = steering_vector_ula(num_rx_ant, angle_deg, self.d_over_lambda) / np.sqrt(num_rx_ant)

        tx_field = np.einsum("t,tnk->nk", a_tx, x)
        y_bf = np.einsum("r,rnk->nk", np.conj(a_rx), y)

        valid = valid_mask & (np.abs(tx_field) > 1e-8)

        per_sc = np.zeros(num_subcarriers, dtype=np.complex128)
        counts = np.zeros(num_subcarriers, dtype=int)

        for sym in range(valid_mask.shape[0]):
            for sc in range(num_subcarriers):
                if valid[sym, sc]:
                    per_sc[sc] += y_bf[sym, sc] / tx_field[sym, sc]
                    counts[sc] += 1

        use_sc = counts > 0
        per_sc[use_sc] /= counts[use_sc]

        return per_sc, use_sc

    def _delay_profile(
        self,
        per_sc: np.ndarray,
        f_k: np.ndarray,
        range_grid: np.ndarray,
    ) -> np.ndarray:
        """Compute delay profile from per-subcarrier channel."""
        tau_grid = 2 * range_grid / C
        steering = np.exp(-1j * 2 * np.pi * tau_grid[:, None] * f_k[None, :])

        profile = np.zeros_like(range_grid, dtype=float)
        for r_idx in range(len(range_grid)):
            profile[r_idx] = np.abs(np.vdot(steering[r_idx], per_sc))

        return profile

    def estimate_music_after_range(
        self,
        y: np.ndarray,
        x: np.ndarray,
        valid_mask: np.ndarray,
        f_k: np.ndarray,
        range_est_m: float,
        tx_angle_est_deg: float,
        angle_grid_deg: np.ndarray,
        num_sources: int = 1,
    ) -> tuple[float, np.ndarray]:
        """Estimate DOA using MUSIC after range compensation.

        First compensates for range-induced phase shift, then
        computes MUSIC spectrum from residual covariance.

        Args:
            y: Received signal
            x: Transmitted signal
            valid_mask: Valid RE mask
            f_k: Subcarrier frequencies
            range_est_m: Estimated range
            tx_angle_est_deg: Estimated transmit angle
            angle_grid_deg: Angle search grid
            num_sources: Number of sources

        Returns:
            doa_est: Estimated DOA in degrees
            spectrum: MUSIC spectrum at angle grid
        """
        num_tx_ant = x.shape[0]
        num_rx_ant = y.shape[0]

        a_tx = steering_vector_ula(num_tx_ant, tx_angle_est_deg, self.d_over_lambda) / np.sqrt(num_tx_ant)
        tx_field = np.einsum("t,tnk->nk", a_tx, x)

        tau_hat = 2 * range_est_m / C
        h_range_hat = np.exp(-1j * 2 * np.pi * f_k * tau_hat)

        snapshots = []
        num_symbols, num_subcarriers = valid_mask.shape

        for sym in range(num_symbols):
            for sc in range(num_subcarriers):
                if valid_mask[sym, sc] and abs(tx_field[sym, sc]) > 1e-8:
                    denom = tx_field[sym, sc] * h_range_hat[sc]
                    snapshots.append(y[:, sym, sc] / denom)

        snapshots = np.asarray(snapshots, dtype=np.complex128)

        if snapshots.shape[0] < num_rx_ant:
            spectrum = np.zeros_like(angle_grid_deg, dtype=float)
            return float(angle_grid_deg[0]), spectrum

        cov = snapshots @ snapshots.conj().T / max(snapshots.shape[0], 1)

        return music_spectrum_from_cov(cov, num_rx_ant, num_sources, angle_grid_deg, self.d_over_lambda)


def delay_music_spectrum(
    y: np.ndarray,
    x: np.ndarray,
    valid_mask: np.ndarray,
    f_k: np.ndarray,
    range_grid: np.ndarray,
    angle_deg: float,
    num_sources: int = 1,
) -> np.ndarray:
    """Delay-domain MUSIC for range estimation.

    Computes MUSIC spectrum in the delay domain after
    beamforming toward given angle.

    Args:
        y: Received signal [num_rx_ant, num_symbols, num_subcarriers]
        x: Transmitted signal [num_tx_ant, num_symbols, num_subcarriers]
        valid_mask: Valid RE mask
        f_k: Subcarrier frequencies
        range_grid: Range search grid
        angle_deg: Beamforming angle
        num_sources: Number of sources

    Returns:
        MUSIC spectrum at range grid
    """
    num_rx_ant = y.shape[0]
    num_tx_ant = x.shape[0]

    a_tx = steering_vector_ula(num_tx_ant, angle_deg) / np.sqrt(num_tx_ant)
    a_rx = steering_vector_ula(num_rx_ant, angle_deg) / np.sqrt(num_rx_ant)

    tx_field = np.einsum("t,tnk->nk", a_tx, x)
    y_bf = np.einsum("r,rnk->nk", np.conj(a_rx), y)

    valid = valid_mask & (np.abs(tx_field) > 1e-8)

    per_sc = np.zeros(f_k.shape[0], dtype=np.complex128)
    counts = np.zeros(f_k.shape[0], dtype=int)

    for sym in range(valid_mask.shape[0]):
        for sc in range(f_k.shape[0]):
            if valid[sym, sc]:
                per_sc[sc] += y_bf[sym, sc] / tx_field[sym, sc]
                counts[sc] += 1

    use_sc = counts > 0
    per_sc[use_sc] /= counts[use_sc]

    h_mat = np.zeros((valid_mask.shape[0], f_k.shape[0]), dtype=np.complex128)
    sc_idx = 0
    for sym in range(valid_mask.shape[0]):
        for sc in range(f_k.shape[0]):
            if valid[sym, sc]:
                h_mat[sym, sc] = per_sc[sc]
                sc_idx += 1

    cov = h_mat @ h_mat.conj().T / max(h_mat.shape[0] * h_mat.shape[1], 1)

    tau_grid = 2 * range_grid / C

    steering_matrix = np.zeros((len(range_grid), cov.shape[0], f_k.shape[0]), dtype=np.complex128)
    for r_idx, r in enumerate(range_grid):
        for s_idx, f in enumerate(f_k):
            steering_matrix[r_idx, :, s_idx] = np.exp(-1j * 2 * np.pi * f * tau_grid[r_idx])

    spectrum = np.zeros(len(range_grid), dtype=float)

    for r_idx in range(len(range_grid)):
        a_delay = steering_matrix[r_idx].flatten()
        spectrum[r_idx] = 1.0 / max(np.abs(a_delay.conj() @ cov @ a_delay), 1e-12)

    return spectrum


def coherent_music_spectrum(
    cov: np.ndarray,
    num_elements: int,
    num_sources: int,
    angle_grid_deg: np.ndarray,
    d_over_lambda: float = 0.5,
) -> np.ndarray:
    """Coherent MUSIC spectrum for multipath suppression.

    Uses coherent averaging of subarray covariances
    to suppress incoherent multipath components.

    Args:
        cov: Full covariance matrix
        num_elements: Number of antenna elements
        num_sources: Number of sources
        angle_grid_deg: Angle search grid
        d_over_lambda: Element spacing

    Returns:
        MUSIC spectrum at angle grid
    """
    from .robust_doa import SpatialSmoothing

    smoother = SpatialSmoothing(num_elements)
    cov_smooth = smoother.apply(cov)

    return music_spectrum_from_cov(cov_smooth, smoother.subarray_size, num_sources, angle_grid_deg, d_over_lambda)


def music_spectrum_from_cov(
    cov: np.ndarray,
    num_elements: int,
    num_sources: int,
    angle_grid_deg: np.ndarray,
    d_over_lambda: float = 0.5,
) -> tuple[float, np.ndarray]:
    """Compute MUSIC spectrum from covariance matrix.

    Args:
        cov: Covariance matrix
        num_elements: Number of antenna elements
        num_sources: Number of sources
        angle_grid_deg: Angle search grid
        d_over_lambda: Element spacing

    Returns:
        best_angle: Angle at spectrum peak
        spectrum: MUSIC spectrum at angle grid
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)
    noise_subspace = eigvecs[:, idx[: max(0, num_elements - num_sources)]]

    spectrum = np.zeros_like(angle_grid_deg, dtype=float)

    for i, ang in enumerate(angle_grid_deg):
        a = steering_vector_ula(num_elements, float(ang), d_over_lambda)
        denom = np.linalg.norm(noise_subspace.conj().T @ a) ** 2
        spectrum[i] = 1.0 / max(denom, 1e-12)

    best_angle = float(angle_grid_deg[int(np.argmax(spectrum))])
    return best_angle, spectrum