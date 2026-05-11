"""Robust DOA algorithms for multipath environments.

This module provides:
1. SpatialSmoothing: Spatial smoothing for decorrelating multipath
2. RootMUSIC: Root-MUSIC for improved resolution
3. BeamspaceMUSIC: Beamspace MUSIC for reduced complexity
4. MUSICWithRefinement: Two-step MUSIC with refinement
5. multipath_robust_music: Multipath-robust MUSIC estimator
6. esprit_with_spatial_smoothing: ESPRIT with spatial smoothing
7. estimate_num_sources_aic/md: Number of source estimation

These algorithms are designed to handle:
- Coherent multipath (delays < correlation time)
- Correlated signals from same direction
- Limited snapshots
- Unknown number of sources

Reference: DOA estimation tutorial + 3GPP channel models
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, svd


def steering_vector_ula(
    num_elements: int,
    angle_deg: float,
    d_over_lambda: float = 0.5,
) -> np.ndarray:
    """Compute ULA steering vector."""
    theta = np.deg2rad(angle_deg)
    n = np.arange(num_elements)
    return np.exp(-1j * 2 * np.pi * d_over_lambda * n * np.sin(theta))


class SpatialSmoothing:
    """Spatial smoothing for decorrelating coherent multipath.

    Spatial smoothing splits the array into overlapping subarrays
    and averages their covariance matrices. This decorrelates
    coherent signals (multipath) that would otherwise degrade
    subspace methods like MUSIC.

    Example:
        >>> smoother = SpatialSmoothing(num_elements=8, subarray_size=6)
        >>> cov_smoothed = smoother.apply(cov_matrix)
        >>> spectrum = music_spectrum(cov_smoothed, ...)
    """

    def __init__(
        self,
        num_elements: int,
        subarray_size: int | None = None,
    ):
        """Initialize spatial smoother.

        Args:
            num_elements: Total number of antenna elements
            subarray_size: Size of each subarray (default: num_elements * 2 // 3)
        """
        self.num_elements = num_elements

        if subarray_size is None:
            subarray_size = int(num_elements * 2 / 3)

        if subarray_size > num_elements:
            subarray_size = num_elements

        self.subarray_size = subarray_size
        self.num_subarrays = num_elements - subarray_size + 1

        if self.num_subarrays < 1:
            self.num_subarrays = 1
            self.subarray_size = num_elements

    def apply(self, cov: np.ndarray) -> np.ndarray:
        """Apply spatial smoothing to covariance matrix.

        Args:
            cov: Full covariance matrix [num_elements, num_elements]

        Returns:
            Smoothed covariance matrix [subarray_size, subarray_size]
        """
        m = self.subarray_size
        l = self.num_subarrays

        cov_smooth = np.zeros((m, m), dtype=np.complex128)

        for i in range(l):
            sub_cov = cov[i : i + m, i : i + m]
            cov_smooth += sub_cov

        cov_smooth /= l
        return cov_smooth

    def apply_full(self, cov: np.ndarray) -> np.ndarray:
        """Apply spatial smoothing preserving full array size.

        This returns a smoothed covariance matrix of the same size
        as the input, suitable for beamspace processing.

        Args:
            cov: Full covariance matrix

        Returns:
            Smoothed covariance matrix [num_elements, num_elements]
        """
        m = self.subarray_size
        l = self.num_subarrays

        cov_smooth = np.zeros_like(cov)
        for i in range(l):
            sub_cov = cov[i : i + m, i : i + m]
            for ii in range(m):
                for jj in range(m):
                    cov_smooth[i + ii, i + jj] += sub_cov[ii, jj] / l

        return cov_smooth


class RootMUSIC:
    """Root-MUSIC for improved DOA resolution.

    Root-MUSIC replaces the spectral search in conventional MUSIC
    with polynomial root-finding, providing better resolution
    and avoiding spectral discretization errors.

    Reference: Barabell, A.J. "Improvement of resolution capability of
    eigenstructure methods via a novel approach", 1983.
    """

    def __init__(self, num_elements: int, num_sources: int):
        """Initialize Root-MUSIC.

        Args:
            num_elements: Number of antenna elements
            num_sources: Number of source signals
        """
        self.num_elements = num_elements
        self.num_sources = num_sources

    def estimate(self, cov: np.ndarray) -> np.ndarray:
        """Estimate DOAs using Root-MUSIC.

        Args:
            cov: Covariance matrix [num_elements, num_elements]

        Returns:
            DOA estimates in degrees (sorted)
        """
        m = self.num_elements
        p = self.num_sources

        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]

        en = eigvecs[:, : max(0, m - p)]

        c_mat = en @ np.conj(en).T

        coefficients = np.zeros(2 * m - 1, dtype=np.complex128)
        for k in range(-(m - 1), m):
            coeff = 0.0 + 0.0j
            for i in range(m):
                j = i + k
                if 0 <= j < m:
                    coeff += c_mat[i, j]
                coefficients[k + m - 1] = coeff

        roots = np.roots(coefficients)

        unit_roots = []
        for root in roots:
            if np.abs(np.abs(root) - 1.0) < 0.1:
                unit_roots.append(root)

        angles = []
        for root in unit_roots:
            angle = np.angle(root)
            theta = np.arcsin(np.clip(-angle / (2 * np.pi * 0.5), -1.0, 1.0))
            angles.append(np.rad2deg(theta))

        angles = np.array(sorted(angles))
        return angles


class BeamspaceMUSIC:
    """Beamspace MUSIC for reduced complexity.

    Beamspace processing transforms the element-space covariance
    to beam space, reducing complexity and improving robustness
    to array imperfections.
    """

    def __init__(
        self,
        num_elements: int,
        num_beams: int,
        num_sources: int,
    ):
        """Initialize Beamspace MUSIC.

        Args:
            num_elements: Number of antenna elements
            num_beams: Number of beams in beamspace
            num_sources: Number of source signals
        """
        self.num_elements = num_elements
        self.num_beams = min(num_beams, num_elements)
        self.num_sources = num_sources

        self._build_beamformer()

    def _build_beamformer(self) -> None:
        """Build DFT beamformer matrix."""
        n = self.num_elements
        k = self.num_beams

        half_k = k // 2
        beam_indices = np.arange(-half_k, half_k)

        self.beam_matrix = np.zeros((n, k), dtype=np.complex128)
        for i, b in enumerate(beam_indices):
            angle = b * np.pi / n
            for m in range(n):
                self.beam_matrix[m, i] = np.exp(1j * m * angle) / np.sqrt(n)

    def spectrum(self, cov: np.ndarray) -> np.ndarray:
        """Compute beamspace MUSIC spectrum.

        Args:
            cov: Element-space covariance matrix

        Returns:
            Spectrum values at angle grid
        """
        cov_beam = np.conj(self.beam_matrix).T @ cov @ self.beam_matrix

        eigvals, eigvecs = np.linalg.eigh(cov_beam)
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]

        en_beam = eigvecs[:, : max(0, self.num_beams - self.num_sources)]

        angle_grid_deg = np.linspace(-90, 90, 361)
        spectrum = np.zeros_like(angle_grid_deg, dtype=float)

        for i, ang in enumerate(angle_grid_deg):
            a = steering_vector_ula(self.num_elements, float(ang))
            a_beam = np.conj(self.beam_matrix).T @ a
            denom = np.linalg.norm(en_beam.conj().T @ a_beam) ** 2
            spectrum[i] = 1.0 / max(denom, 1e-12)

        return angle_grid_deg, spectrum


class MUSICWithRefinement:
    """Two-step MUSIC with iterative refinement.

    First step: Coarse search with coarse grid
    Second step: Refinement around initial peaks
    This provides both efficiency and precision.
    """

    def __init__(
        self,
        num_elements: int,
        num_sources: int,
        coarse_resolution: float = 1.0,
        fine_resolution: float = 0.1,
    ):
        """Initialize refinement MUSIC.

        Args:
            num_elements: Number of antenna elements
            num_sources: Number of source signals
            coarse_resolution: Coarse search resolution in degrees
            fine_resolution: Fine search resolution in degrees
        """
        self.num_elements = num_elements
        self.num_sources = num_sources
        self.coarse_resolution = coarse_resolution
        self.fine_resolution = fine_resolution

    def estimate(self, cov: np.ndarray) -> np.ndarray:
        """Estimate DOAs with two-step refinement.

        Args:
            cov: Covariance matrix

        Returns:
            DOA estimates in degrees
        """
        coarse_grid = np.arange(-90, 90, self.coarse_resolution)

        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)
        noise_subspace = eigvecs[:, idx[: max(0, self.num_elements - self.num_sources)]]

        coarse_spectrum = np.zeros_like(coarse_grid, dtype=float)
        for i, ang in enumerate(coarse_grid):
            a = steering_vector_ula(self.num_elements, float(ang))
            denom = np.linalg.norm(noise_subspace.conj().T @ a) ** 2
            coarse_spectrum[i] = 1.0 / max(denom, 1e-12)

        peak_indices = self._find_peaks(coarse_spectrum, self.num_sources)

        angles = []
        for peak_idx in peak_indices:
            coarse_peak = coarse_grid[peak_idx]

            fine_grid = np.arange(
                max(-90, coarse_peak - self.coarse_resolution),
                min(90, coarse_peak + self.coarse_resolution) + 1e-9,
                self.fine_resolution,
            )

            fine_spectrum = np.zeros_like(fine_grid, dtype=float)
            for i, ang in enumerate(fine_grid):
                a = steering_vector_ula(self.num_elements, float(ang))
                denom = np.linalg.norm(noise_subspace.conj().T @ a) ** 2
                fine_spectrum[i] = 1.0 / max(denom, 1e-12)

            fine_peak_idx = np.argmax(fine_spectrum)
            angles.append(fine_grid[fine_peak_idx])

        return np.sort(np.array(angles, dtype=float))

    def _find_peaks(self, spectrum: np.ndarray, num_peaks: int) -> list[int]:
        """Find peaks in spectrum with non-maximum suppression."""
        peak_indices = []
        sorted_indices = np.argsort(spectrum)[::-1]

        for idx in sorted_indices:
            if len(peak_indices) >= num_peaks:
                break

            is_valid = True
            for prev_idx in peak_indices:
                if abs(idx - prev_idx) < 3:
                    is_valid = False
                    break

            if is_valid:
                peak_indices.append(int(idx))

        return peak_indices


def multipath_robust_music(
    cov: np.ndarray,
    num_elements: int,
    num_sources: int,
    angle_grid_deg: np.ndarray,
    d_over_lambda: float = 0.5,
    smoothing_size: int | None = None,
) -> np.ndarray:
    """Robust MUSIC estimator for multipath environments.

    Applies spatial smoothing to handle coherent multipath
    before running MUSIC spectrum estimation.

    Args:
        cov: Covariance matrix
        num_elements: Number of antenna elements
        num_sources: Number of source signals
        angle_grid_deg: Search grid for spectrum
        d_over_lambda: Element spacing
        smoothing_size: Subarray size for smoothing

    Returns:
        MUSIC spectrum at angle grid
    """
    if smoothing_size is None:
        smoothing_size = int(num_elements * 2 / 3)

    smoother = SpatialSmoothing(num_elements, smoothing_size)
    cov_smooth = smoother.apply(cov)

    eigvals, eigvecs = np.linalg.eigh(cov_smooth)
    idx = np.argsort(eigvals)
    noise_subspace = eigvecs[:, idx[: max(0, smoothing_size - num_sources)]]

    spectrum = np.zeros_like(angle_grid_deg, dtype=float)
    for i, ang in enumerate(angle_grid_deg):
        a = steering_vector_ula(smoothing_size, float(ang), d_over_lambda)
        denom = np.linalg.norm(noise_subspace.conj().T @ a) ** 2
        spectrum[i] = 1.0 / max(denom, 1e-12)

    return spectrum


def esprit_with_spatial_smoothing(
    cov: np.ndarray,
    num_elements: int,
    num_sources: int,
    d_over_lambda: float = 0.5,
    smoothing_size: int | None = None,
) -> np.ndarray:
    """ESPRIT with spatial smoothing for multipath.

    Args:
        cov: Covariance matrix
        num_elements: Number of antenna elements
        num_sources: Number of source signals
        d_over_lambda: Element spacing
        smoothing_size: Subarray size for smoothing

    Returns:
        DOA estimates in degrees
    """
    if smoothing_size is None:
        smoothing_size = int(num_elements * 2 / 3)

    smoother = SpatialSmoothing(num_elements, smoothing_size)
    cov_smooth = smoother.apply(cov)

    eigvals, eigvecs = np.linalg.eigh(cov_smooth)
    order = np.argsort(eigvals)[::-1]
    es = eigvecs[:, order[:num_sources]]

    es1 = es[:-1, :]
    es2 = es[1:, :]

    psi = np.linalg.pinv(es1) @ es2
    phi = np.linalg.eigvals(psi)

    spatial_freq = -np.angle(phi) / (2 * np.pi * d_over_lambda)
    spatial_freq = np.clip(spatial_freq.real, -1.0, 1.0)
    angles = np.rad2deg(np.arcsin(spatial_freq))

    return np.sort(angles.astype(float))


def estimate_num_sources_aic(cov: np.ndarray) -> int:
    """Estimate number of sources using AIC criterion.

    Args:
        cov: Covariance matrix [M, M]

    Returns:
        Estimated number of sources
    """
    m = cov.shape[0]

    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)[::-1]

    eigvals = np.maximum(eigvals, 1e-12)

    log_sum = np.zeros(m)
    for k in range(m):
        if k < m - 1:
            numerator = np.prod(eigvals[k:m])
            denominator = ((np.sum(eigvals[k:m]) / (m - k)) ** (m - k))
            if denominator > 0 and numerator > 0:
                log_sum[k] = (m - k) * np.log(numerator / denominator)

    aic = np.zeros(m)
    for k in range(m):
        aic[k] = 2 * k * m + 2 * log_sum[k]

    return int(np.argmin(aic))


def estimate_num_sources_md(cov: np.ndarray) -> int:
    """Estimate number of sources using MDL criterion.

    Args:
        cov: Covariance matrix [M, M]

    Returns:
        Estimated number of sources
    """
    m = cov.shape[0]

    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)[::-1]

    eigvals = np.maximum(eigvals, 1e-12)

    log_sum = np.zeros(m)
    for k in range(m):
        if k < m - 1:
            numerator = np.prod(eigvals[k:m])
            denominator = ((np.sum(eigvals[k:m]) / (m - k)) ** (m - k))
            if denominator > 0 and numerator > 0:
                log_sum[k] = (m - k) * np.log(numerator / denominator)

    n_samples = m

    mdl = np.zeros(m)
    for k in range(m):
        mdl[k] = k * m * np.log(n_samples) + 2 * log_sum[k]

    return int(np.argmin(mdl))