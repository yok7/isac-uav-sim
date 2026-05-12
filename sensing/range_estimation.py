"""Range estimation via coarse-to-fine delay profile search."""

from __future__ import annotations

import numpy as np

from doa.classical import steering_vector_ula


C = 299_792_458.0


def per_subcarrier_channel(
    y: np.ndarray,
    x: np.ndarray,
    valid: np.ndarray,
    angle_deg: float,
    num_tx_ant: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Beamform toward angle and compute per-subcarrier channel.

    Args:
        y: Received signal [num_rx_ant, num_symbols, num_subcarriers]
        x: Transmitted signal [num_tx_ant, num_symbols, num_subcarriers]
        valid: Valid RE mask [num_symbols, num_subcarriers]
        angle_deg: Beamforming angle
        num_tx_ant: Number of TX antennas

    Returns:
        per_sc: Per-subcarrier channel estimate [num_subcarriers]
        use_sc: Valid subcarrier mask
    """
    num_rx_ant = y.shape[0]
    num_subcarriers = x.shape[-1]

    a_tx = steering_vector_ula(num_tx_ant, angle_deg) / np.sqrt(num_tx_ant)
    a_rx = steering_vector_ula(num_rx_ant, angle_deg) / np.sqrt(num_rx_ant)

    # TX matched filter + RX beamforming
    tx_field = np.einsum("t,tnk->nk", a_tx, x)
    y_bf = np.einsum("r,rnk->nk", np.conj(a_rx), y)

    valid_mask = valid & (np.abs(tx_field) > 1e-8)

    per_sc = np.zeros(num_subcarriers, dtype=np.complex128)
    counts = np.zeros(num_subcarriers, dtype=int)

    for sym in range(valid.shape[0]):
        for sc in range(num_subcarriers):
            if valid_mask[sym, sc]:
                per_sc[sc] += y_bf[sym, sc] / tx_field[sym, sc]
                counts[sc] += 1

    use_sc = counts > 0
    per_sc[use_sc] /= counts[use_sc]

    return per_sc, use_sc


def delay_profile(
    per_sc: np.ndarray,
    f_k: np.ndarray,
    range_grid: np.ndarray,
) -> np.ndarray:
    """Compute matched-filter delay profile.

    Args:
        per_sc: Per-subcarrier channel [num_subcarriers]
        f_k: Subcarrier frequencies [num_subcarriers]
        range_grid: Range search grid [num_ranges]

    Returns:
        profile: Delay profile [num_ranges]
    """
    tau_grid = 2 * range_grid / C
    profile = np.zeros_like(range_grid, dtype=float)

    for r_idx in range(len(range_grid)):
        steering = np.exp(-1j * 2 * np.pi * tau_grid[r_idx] * f_k)
        profile[r_idx] = np.abs(np.vdot(steering, per_sc))

    return profile


def coarse_to_fine_range_angle(
    y: np.ndarray,
    x: np.ndarray,
    valid: np.ndarray,
    f_k: np.ndarray,
    num_tx_ant: int,
    coarse_resolution: float = 2.0,
    fine_resolution: float = 0.25,
    range_min: float = 0.0,
    range_max: float = 300.0,
    angle_min: float = -60.0,
    angle_max: float = 60.0,
) -> tuple[float, float]:
    """Coarse-to-fine joint range and angle estimation.

    Returns:
        best_range: Estimated range in meters
        best_angle: Estimated angle in degrees
    """
    coarse_ranges = np.arange(range_min, range_max + 1e-9, coarse_resolution)
    coarse_angles = np.arange(angle_min, angle_max + 1e-9, coarse_resolution)

    best_score = -np.inf
    best_range = coarse_ranges[0]
    best_angle = coarse_angles[0]

    for ang in coarse_angles:
        per_sc, _ = per_subcarrier_channel(y, x, valid, float(ang), num_tx_ant)
        profile = delay_profile(per_sc, f_k, coarse_ranges)
        peak_idx = np.argmax(profile)
        score = float(profile[peak_idx])

        if score > best_score:
            best_score = score
            best_range = float(coarse_ranges[peak_idx])
            best_angle = float(ang)

    fine_ranges = np.arange(
        max(range_min, best_range - 2.0),
        min(range_max, best_range + 2.0) + 1e-9,
        fine_resolution,
    )
    fine_angles = np.arange(
        max(angle_min, best_angle - 2.0),
        min(angle_max, best_angle + 2.0) + 1e-9,
        fine_resolution,
    )

    best_score = -np.inf
    for ang in fine_angles:
        per_sc, _ = per_subcarrier_channel(y, x, valid, float(ang), num_tx_ant)
        profile = delay_profile(per_sc, f_k, fine_ranges)
        peak_idx = np.argmax(profile)
        score = float(profile[peak_idx])

        if score > best_score:
            best_score = score
            best_range = float(fine_ranges[peak_idx])
            best_angle = float(ang)

    return best_range, best_angle