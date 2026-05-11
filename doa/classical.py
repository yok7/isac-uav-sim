"""Classical DOA algorithms for ULA."""

from __future__ import annotations

import numpy as np


def steering_vector_ula(num_elements: int, angle_deg: float, d_over_lambda: float = 0.5) -> np.ndarray:
    """Return ULA steering vector a(theta)."""
    theta = np.deg2rad(angle_deg)
    n = np.arange(num_elements)
    # Use negative phase progression so positive angle maps to positive DOA in our convention.
    return np.exp(-1j * 2 * np.pi * d_over_lambda * n * np.sin(theta))


def beamforming_spectrum(
    cov: np.ndarray,
    num_elements: int,
    angle_grid_deg: np.ndarray,
    d_over_lambda: float = 0.5,
) -> np.ndarray:
    """Conventional beamforming spectrum."""
    p = np.zeros_like(angle_grid_deg, dtype=float)
    for i, ang in enumerate(angle_grid_deg):
        a = steering_vector_ula(num_elements, float(ang), d_over_lambda)
        p[i] = np.real(np.conj(a).T @ cov @ a)
    return p


def music_spectrum(
    cov: np.ndarray,
    num_elements: int,
    num_sources: int,
    angle_grid_deg: np.ndarray,
    d_over_lambda: float = 0.5,
) -> np.ndarray:
    """MUSIC pseudo-spectrum."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)
    noise_subspace = eigvecs[:, idx[: max(0, num_elements - num_sources)]]

    p = np.zeros_like(angle_grid_deg, dtype=float)
    for i, ang in enumerate(angle_grid_deg):
        a = steering_vector_ula(num_elements, float(ang), d_over_lambda)
        denom = np.linalg.norm(np.conj(noise_subspace).T @ a) ** 2
        p[i] = 1.0 / max(denom, 1e-12)
    return p


def estimate_peak_angle(angle_grid_deg: np.ndarray, spectrum: np.ndarray) -> float:
    """Estimate DOA by spectrum peak."""
    return float(angle_grid_deg[int(np.argmax(spectrum))])


def estimate_topk_angles(
    angle_grid_deg: np.ndarray,
    spectrum: np.ndarray,
    num_sources: int,
    min_separation_deg: float = 2.0,
) -> np.ndarray:
    """Estimate multiple DOAs from a spatial spectrum with simple non-maximum suppression."""
    if num_sources <= 0:
        return np.array([], dtype=float)

    idx_sorted = np.argsort(spectrum)[::-1]
    picked: list[float] = []

    for idx in idx_sorted:
        cand = float(angle_grid_deg[idx])
        if all(abs(cand - prev) >= min_separation_deg for prev in picked):
            picked.append(cand)
        if len(picked) >= num_sources:
            break

    picked = sorted(picked)
    return np.asarray(picked, dtype=float)


def beamforming_estimate(
    cov: np.ndarray,
    num_elements: int,
    num_sources: int,
    angle_grid_deg: np.ndarray,
    d_over_lambda: float = 0.5,
) -> np.ndarray:
    """Estimate DOAs with conventional beamforming."""
    spectrum = beamforming_spectrum(cov, num_elements, angle_grid_deg, d_over_lambda)
    return estimate_topk_angles(angle_grid_deg, spectrum, num_sources)


def music_estimate(
    cov: np.ndarray,
    num_elements: int,
    num_sources: int,
    angle_grid_deg: np.ndarray,
    d_over_lambda: float = 0.5,
) -> np.ndarray:
    """Estimate DOAs with MUSIC."""
    spectrum = music_spectrum(cov, num_elements, num_sources, angle_grid_deg, d_over_lambda)
    return estimate_topk_angles(angle_grid_deg, spectrum, num_sources)


def esprit_estimate(
    cov: np.ndarray,
    num_sources: int,
    d_over_lambda: float = 0.5,
) -> np.ndarray:
    """
    Estimate DOAs with ESPRIT for ULA.

    Notes:
    - Assumes shift-invariant ULA and number of sources is known.
    - Uses least-squares rotational invariance relation.
    """
    m = cov.shape[0]
    if num_sources <= 0 or num_sources >= m:
        return np.array([], dtype=float)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    es = eigvecs[:, order[:num_sources]]

    es1 = es[:-1, :]
    es2 = es[1:, :]
    psi = np.linalg.pinv(es1) @ es2
    phi = np.linalg.eigvals(psi)

    # With a(theta)=exp(-j2*pi*d/lambda*n*sin(theta)), phase increment is -2*pi*d*sin(theta).
    spatial_freq = -np.angle(phi) / (2 * np.pi * d_over_lambda)
    spatial_freq = np.clip(spatial_freq.real, -1.0, 1.0)
    angles = np.rad2deg(np.arcsin(spatial_freq))
    return np.sort(angles.astype(float))
