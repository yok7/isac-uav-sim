"""ULA snapshot simulation for low-altitude UAV sensing experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from doa.classical import steering_vector_ula


@dataclass(frozen=True)
class UlaSceneConfig:
    num_elements: int = 8
    d_over_lambda: float = 0.5
    num_snapshots: int = 128
    snr_db: float = 5.0
    target_doas_deg: tuple[float, ...] = (-15.0, 20.0)
    multipath: bool = False
    multipath_atten_db: float = 8.0
    multipath_spread_deg: float = 6.0


def simulate_ula_snapshots(
    cfg: UlaSceneConfig,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate array snapshots and covariance matrix.

    Returns:
        x: complex snapshots with shape [num_snapshots, num_elements]
        cov: sample covariance matrix [num_elements, num_elements]
        true_doas: true LoS target angles used for scoring
    """
    rng = np.random.default_rng(seed)

    m = cfg.num_elements
    n = cfg.num_snapshots
    true_doas = np.asarray(cfg.target_doas_deg, dtype=float)

    x_signal = np.zeros((n, m), dtype=np.complex128)

    for doa in true_doas:
        a = steering_vector_ula(m, float(doa), cfg.d_over_lambda)
        s = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        x_signal += np.outer(s, np.conj(a))

        if cfg.multipath:
            refl_shift = rng.uniform(-cfg.multipath_spread_deg, cfg.multipath_spread_deg)
            refl_doa = float(np.clip(doa + refl_shift, -85.0, 85.0))
            a_refl = steering_vector_ula(m, refl_doa, cfg.d_over_lambda)
            refl_gain = 10.0 ** (-cfg.multipath_atten_db / 20.0)
            s_refl = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
            x_signal += refl_gain * np.outer(s_refl, np.conj(a_refl))

    sig_power = float(np.mean(np.abs(x_signal) ** 2))
    snr_lin = 10.0 ** (cfg.snr_db / 10.0)
    noise_power = sig_power / max(snr_lin, 1e-12)
    noise = np.sqrt(noise_power / 2.0) * (rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m)))

    x = x_signal + noise
    cov = (x.conj().T @ x) / n
    return x, cov, true_doas
