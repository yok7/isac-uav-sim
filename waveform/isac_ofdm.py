"""ISAC OFDM waveform and resource allocation helpers for Phase-2 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc


@dataclass(frozen=True)
class IsacOfdmConfig:
    num_subcarriers: int = 128
    num_symbols: int = 14
    subcarrier_spacing_khz: float = 30.0
    cp_ratio: float = 0.125
    modulation_order: int = 4
    pilot_symbol_indices: tuple[int, ...] = (2, 11)
    pilot_density: float = 0.1
    comm_ratio: float = 0.6


def _build_masks(cfg: IsacOfdmConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    total_re = cfg.num_symbols * cfg.num_subcarriers
    all_idx = np.arange(total_re)

    pilot_mask = np.zeros(total_re, dtype=bool)
    pilot_syms = [s for s in cfg.pilot_symbol_indices if 0 <= s < cfg.num_symbols]
    if pilot_syms:
        step = max(1, int(round(1.0 / max(cfg.pilot_density, 1e-3))))
        for sym in pilot_syms:
            start = sym * cfg.num_subcarriers
            idx = np.arange(start, start + cfg.num_subcarriers, step)
            pilot_mask[idx] = True

    remain_idx = all_idx[~pilot_mask]
    comm_count = int(np.clip(cfg.comm_ratio, 0.0, 1.0) * remain_idx.size)
    comm_idx = rng.choice(remain_idx, size=comm_count, replace=False) if comm_count > 0 else np.array([], dtype=int)

    comm_mask = np.zeros(total_re, dtype=bool)
    comm_mask[comm_idx] = True

    sensing_mask = ~(pilot_mask | comm_mask)

    return {
        "pilot": pilot_mask.reshape(cfg.num_symbols, cfg.num_subcarriers),
        "comm": comm_mask.reshape(cfg.num_symbols, cfg.num_subcarriers),
        "sensing": sensing_mask.reshape(cfg.num_symbols, cfg.num_subcarriers),
    }


def _generate_qpsk(n: int, rng: np.random.Generator) -> np.ndarray:
    bits = rng.integers(0, 2, size=(n, 2))
    re = 2 * bits[:, 0] - 1
    im = 2 * bits[:, 1] - 1
    return (re + 1j * im) / np.sqrt(2)


def generate_isac_ofdm_frame(
    cfg: IsacOfdmConfig,
    seed: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Generate a simplified ISAC OFDM frame and its resource masks."""
    rng = np.random.default_rng(seed)
    masks = _build_masks(cfg, rng)

    grid = np.zeros((cfg.num_symbols, cfg.num_subcarriers), dtype=np.complex128)
    comm_n = int(masks["comm"].sum())
    if comm_n > 0:
        grid[masks["comm"]] = _generate_qpsk(comm_n, rng)

    grid[masks["pilot"]] = 1.0 + 0j

    cp_len = int(round(cfg.cp_ratio * cfg.num_subcarriers))
    td_symbols = np.fft.ifft(grid, axis=1)
    with_cp = np.concatenate([td_symbols[:, -cp_len:], td_symbols], axis=1) if cp_len > 0 else td_symbols
    waveform = with_cp.reshape(-1)

    total_re = cfg.num_subcarriers * cfg.num_symbols
    comm_fraction = float(masks["comm"].sum() / total_re)
    pilot_fraction = float(masks["pilot"].sum() / total_re)
    sensing_fraction = float(masks["sensing"].sum() / total_re)

    return {
        "resource_grid": grid,
        "time_waveform": waveform,
        "masks": masks,
        "comm_fraction": comm_fraction,
        "pilot_fraction": pilot_fraction,
        "sensing_fraction": sensing_fraction,
    }


def effective_comm_rate_bpshz(
    snr_db: float,
    comm_fraction: float,
    pilot_fraction: float,
    modulation_order: int = 4,
) -> float:
    """Proxy communication rate for fixed total resources."""
    snr_lin = 10.0 ** (snr_db / 10.0)
    shannon = np.log2(1.0 + snr_lin)
    mod_ceiling = np.log2(max(modulation_order, 2))
    spectral_eff = min(shannon, mod_ceiling)
    payload_factor = max(0.0, 1.0 - 0.5 * pilot_fraction)
    return float(max(0.0, comm_fraction) * payload_factor * spectral_eff)


def ber_proxy_qpsk(
    snr_db: float,
    pilot_fraction: float,
    sensing_fraction: float,
) -> float:
    """Approximate BER proxy for QPSK under varying pilot/sensing resource pressure."""
    snr_lin = 10.0 ** (snr_db / 10.0)
    # Less pilot generally hurts communication channel estimation; more sensing resource also reduces comm robustness.
    penalty = 1.0 + 0.7 * max(0.0, 0.12 - pilot_fraction) + 0.25 * max(0.0, sensing_fraction - 0.3)
    eff_snr = snr_lin / penalty
    ber = 0.5 * erfc(np.sqrt(max(eff_snr, 1e-12)))
    return float(np.clip(ber, 1e-7, 0.5))
