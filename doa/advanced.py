"""
Advanced DOA baseline surrogates for Phase-2.

Important:
- These are lightweight, reproducible surrogates inspired by the target papers.
- They are not official reproductions of TransDOA / OGFVBI.
"""

from __future__ import annotations

import numpy as np

from doa.classical import estimate_topk_angles, music_spectrum, steering_vector_ula


def _forward_backward_averaging(cov: np.ndarray) -> np.ndarray:
    m = cov.shape[0]
    j = np.fliplr(np.eye(m))
    return 0.5 * (cov + j @ cov.conj() @ j)


def transdoa_surrogate_spectrum(
    cov: np.ndarray,
    num_elements: int,
    num_sources: int,
    angle_grid_deg: np.ndarray,
    d_over_lambda: float = 0.5,
    shrinkage: float = 0.18,
) -> np.ndarray:
    """
    Robust-MUSIC style surrogate for array-imperfection robustness.

    Pipeline:
    1) covariance shrinkage regularization
    2) forward-backward averaging
    3) MUSIC pseudo-spectrum
    """
    tr = float(np.trace(cov).real)
    eye = np.eye(num_elements)
    cov_reg = (1.0 - shrinkage) * cov + shrinkage * (tr / num_elements) * eye
    cov_fb = _forward_backward_averaging(cov_reg)
    return music_spectrum(cov_fb, num_elements, num_sources, angle_grid_deg, d_over_lambda)


def transdoa_surrogate_estimate(
    cov: np.ndarray,
    num_elements: int,
    num_sources: int,
    angle_grid_deg: np.ndarray,
    d_over_lambda: float = 0.5,
) -> np.ndarray:
    spectrum = transdoa_surrogate_spectrum(
        cov=cov,
        num_elements=num_elements,
        num_sources=num_sources,
        angle_grid_deg=angle_grid_deg,
        d_over_lambda=d_over_lambda,
    )
    return estimate_topk_angles(angle_grid_deg, spectrum, num_sources)


def ogfvbi_surrogate_spectrum(
    cov: np.ndarray,
    num_elements: int,
    angle_grid_deg: np.ndarray,
    d_over_lambda: float = 0.5,
    max_iters: int = 6,
    reg: float = 1e-2,
) -> np.ndarray:
    """
    Off-grid sparse Bayesian inspired surrogate.

    Uses iterative reweighted Capon-like updates to emphasize sparse peaks,
    then can be refined by quadratic interpolation in the estimate function.
    """
    eye = np.eye(num_elements)
    cov_inv = np.linalg.inv(cov + reg * np.trace(cov).real / num_elements * eye)

    a_mat = np.stack(
        [steering_vector_ula(num_elements, float(ang), d_over_lambda) for ang in angle_grid_deg],
        axis=1,
    )

    denom = np.real(np.sum(np.conj(a_mat) * (cov_inv @ a_mat), axis=0))
    p = 1.0 / np.maximum(denom, 1e-12)
    p = p / np.max(p)

    for _ in range(max_iters):
        w = 1.0 / np.maximum(p, 1e-8)
        p = 1.0 / np.maximum(denom * w, 1e-10)
        p = p / np.max(p)

    return p


def _quadratic_peak_refine(
    x_grid: np.ndarray,
    y: np.ndarray,
    idx: int,
) -> float:
    """Refine peak location via local quadratic interpolation."""
    if idx <= 0 or idx >= len(x_grid) - 1:
        return float(x_grid[idx])

    x1, x2, x3 = x_grid[idx - 1 : idx + 2]
    y1, y2, y3 = y[idx - 1 : idx + 2]

    denom = (y1 - 2.0 * y2 + y3)
    if abs(denom) < 1e-12:
        return float(x2)

    delta = 0.5 * (y1 - y3) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    step = float(x2 - x1)
    return float(x2 + delta * step)


def ogfvbi_surrogate_estimate(
    cov: np.ndarray,
    num_elements: int,
    num_sources: int,
    angle_grid_deg: np.ndarray,
    d_over_lambda: float = 0.5,
) -> np.ndarray:
    spectrum = ogfvbi_surrogate_spectrum(cov, num_elements, angle_grid_deg, d_over_lambda)
    coarse = estimate_topk_angles(angle_grid_deg, spectrum, num_sources)

    refined: list[float] = []
    for c in coarse:
        idx = int(np.argmin(np.abs(angle_grid_deg - c)))
        refined.append(_quadratic_peak_refine(angle_grid_deg, spectrum, idx))

    return np.sort(np.asarray(refined, dtype=float))
