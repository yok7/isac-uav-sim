"""End-to-end ISAC simulation with Sionna RT ray-tracing.

Complete simulation pipeline:
  Sionna PUSCH OFDM waveform
  → Sionna RT multipath two-leg echo (BS→UAV→BS)
  → Moving target Doppler phase
  → AWGN receive noise
  → DMRS-based range estimation
  → MUSIC DOA estimation after range compensation
  → RMSE statistics

Usage:
    python isac_end2end_simulation.py --mode full --num-trials 100
    python isac_end2end_simulation.py --compare-channels --num-trials 50

Output: results/end2end/simulation_results.csv with fields:
  - channel_type, max_depth, specular_reflection, diffuse_scattering
  - target_velocity_mps, snr_db, range_rmse_m, doa_rmse_deg, num_trials
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "end2end"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import sys

sys_path = str(PROJECT_ROOT)
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from channel import ChannelModelType, SionnaRTChannel, SionnaRTConfig
from waveform import SionnaWaveformGenerator, IsacWaveformConfig
from doa.classical import music_estimate, steering_vector_ula
from doa.metrics import doa_rmse_deg
try:
    from doa import HyperDOAEstimator, HyperDOAConfig
except ImportError:
    HyperDOAEstimator = None
    HyperDOAConfig = None


C = 299_792_458.0


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class SimulationConfig:
    """Configuration for end-to-end ISAC simulation."""

    mode: str = "full"

    snr_min_db: float = -5.0
    snr_max_db: float = 20.0
    snr_step_db: float = 5.0

    num_trials: int = 30
    num_tx_ant: int = 4
    num_rx_ant: int = 8
    num_pusch_symbols: int = 14
    subcarrier_spacing_khz: float = 30.0
    num_rbs: int = 18

    # Target position (PPT: [120, 45, 35] m)
    bs_position: tuple[float, float, float] = (0.0, 0.0, 25.0)
    uav_position: tuple[float, float, float] = (120.0, 45.0, 35.0)

    # Moving target velocity [vx, vy, vz] m/s (None = static)
    target_velocity_mps: tuple[float, float, float] | None = None

    algorithms: tuple[str, ...] = ("music",)
    output_dir: Path = field(default_factory=lambda: RESULTS_DIR)
    seed: int = 42

    def __post_init__(self):
        self.uav_distance_m = np.sqrt(
            (self.uav_position[0] - self.bs_position[0]) ** 2
            + (self.uav_position[1] - self.bs_position[1]) ** 2
        )
        bearing_rad = np.arctan2(
            self.uav_position[1] - self.bs_position[1],
            self.uav_position[0] - self.bs_position[0],
        )
        self.uav_bearing_deg = float(np.rad2deg(bearing_rad))

    def get_waveform_config(self) -> IsacWaveformConfig:
        return IsacWaveformConfig(
            subcarrier_spacing_khz=self.subcarrier_spacing_khz,
            num_rbs=self.num_rbs,
            num_pusch_symbols=self.num_pusch_symbols,
            num_tx_ant=self.num_tx_ant,
            num_layers=2,
            num_rx_ant=self.num_rx_ant,
        )

    def get_rt_config(self, model_type: ChannelModelType) -> SionnaRTConfig:
        cfg = SionnaRTConfig(
            model_type=model_type,
            carrier_frequency_hz=3.5e9,
            max_depth=3,
            samples_per_src=5000,
            bs_position=self.bs_position,
            uav_height_m=self.uav_position[2],
            uav_distance_m=self.uav_distance_m,
            uav_bearing_deg=self.uav_bearing_deg,
            num_bs_ant=self.num_rx_ant,
            specular_reflection=True,
            diffuse_scattering=True,
        )

        if model_type == ChannelModelType.LOS:
            cfg.max_depth = 0
            cfg.specular_reflection = False
            cfg.diffuse_scattering = False

        return cfg


# ============================================================================
# Results
# ============================================================================


@dataclass
class SimulationResult:
    """Single trial result."""

    channel_type: str
    doa_method: str
    max_depth: int
    specular_reflection: bool
    diffuse_scattering: bool
    target_velocity_mps: float | None
    snr_db: float

    range_rmse_m: float
    range_bias_m: float
    range_std_m: float

    doa_rmse_deg: float
    doa_bias_deg: float
    doa_std_deg: float

    vel_rmse_mps: float | None = None
    vel_bias_mps: float | None = None
    vel_std_mps: float | None = None
    true_radial_velocity_mps: float | None = None

    runtime_ms: float | None = None
    num_trials: int = 1


# ============================================================================
# Range & DOA Estimation
# ============================================================================


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


def estimate_doa_music_after_range(
    y: np.ndarray,
    x: np.ndarray,
    valid: np.ndarray,
    f_k: np.ndarray,
    range_est_m: float,
    tx_angle_est_deg: float,
    num_tx_ant: int,
    num_rx_ant: int,
    angle_grid_deg: np.ndarray,
) -> float:
    """Estimate DOA using MUSIC after range compensation.

    Args:
        y: Received signal [num_rx_ant, num_symbols, num_subcarriers]
        x: Transmitted signal [num_tx_ant, num_symbols, num_subcarriers]
        valid: Valid RE mask
        f_k: Subcarrier frequencies
        range_est_m: Estimated range for phase compensation
        tx_angle_est_deg: Estimated TX angle for steering
        num_tx_ant: Number of TX antennas
        num_rx_ant: Number of RX antennas
        angle_grid_deg: DOA search grid

    Returns:
        doa_est: Estimated DOA in degrees
    """
    a_tx = steering_vector_ula(num_tx_ant, tx_angle_est_deg) / np.sqrt(num_tx_ant)
    tx_field = np.einsum("t,tnk->nk", a_tx, x)

    # Range-induced phase compensation
    tau_hat = 2 * range_est_m / C
    h_range = np.exp(-1j * 2 * np.pi * f_k * tau_hat)

    # Collect snapshots from valid REs
    snapshots = []
    for sym in range(valid.shape[0]):
        for sc in range(valid.shape[1]):
            if valid[sym, sc] and abs(tx_field[sym, sc]) > 1e-8:
                denom = tx_field[sym, sc] * h_range[sc]
                snapshots.append(y[:, sym, sc] / denom)

    snapshots = np.asarray(snapshots, dtype=np.complex128)

    if snapshots.shape[0] < num_rx_ant:
        return float(angle_grid_deg[0])

    # Rx array covariance: [num_rx_ant, num_rx_ant]
    # snapshots shape: [num_snapshots, num_rx_ant]
    # ( snapshots.T @ snapshots.conj() ) -> [num_rx_ant, num_rx_ant]
    cov = snapshots.T @ snapshots.conj() / max(snapshots.shape[0], 1)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # One target: signal subspace dim = 1, rest is noise
    num_sources = 1
    noise_subspace = eigvecs[:, : num_rx_ant - num_sources]

    # MUSIC spectrum (noise subspace)
    spectrum = np.zeros_like(angle_grid_deg, dtype=float)
    for i, ang in enumerate(angle_grid_deg):
        a = steering_vector_ula(num_rx_ant, float(ang))
        denom = np.linalg.norm(noise_subspace.conj().T @ a) ** 2
        spectrum[i] = 1.0 / max(denom, 1e-12)

    return float(angle_grid_deg[np.argmax(spectrum)])


def estimate_radial_velocity_doppler(
    y_frames: np.ndarray,
    valid_mask: np.ndarray,
    range_est_m: float,
    f_k: np.ndarray,
    frame_interval_s: float,
    carrier_frequency_hz: float,
) -> tuple[float, float]:
    """Estimate radial velocity via Doppler FFT across all subcarriers and antennas.

    Strategy: Compute Doppler FFT independently per (symbol, subcarrier)
    and average power spectra across valid REs and RX antennas. This preserves
    subcarrier and spatial diversity while maintaining slow-time coherence.

    Args:
        y_frames: Echo over frames [num_frames, num_rx_ant, num_symbols, num_subcarriers]
        valid_mask: Valid RE mask [num_symbols, num_subcarriers]
        range_est_m: Estimated range from range estimator (m)
        f_k: Subcarrier frequencies [num_subcarriers]
        frame_interval_s: Time between frames (seconds)
        carrier_frequency_hz: Carrier frequency in Hz

    Returns:
        est_velocity_mps: Estimated radial velocity (m/s)
        doppler_hz: Estimated Doppler frequency (Hz)
    """
    num_frames = y_frames.shape[0]
    num_rx_ant = y_frames.shape[1]
    num_symbols = y_frames.shape[2]
    num_subcarriers = y_frames.shape[3]

    # Phase compensation per subcarrier to remove range-induced phase
    tau_est = 2 * range_est_m / C
    phase_comp = np.exp(-1j * 2 * np.pi * tau_est * f_k)  # [num_subcarriers]

    # Accumulate Doppler power spectra: sum |FFT(slow_time)|^2 over all RE × RX
    accumulated_spectrum = np.zeros(num_frames, dtype=float)

    for sym in range(num_symbols):
        for sc in range(num_subcarriers):
            if not valid_mask[sym, sc]:
                continue

            # Collect slow-time vector for this RE across frames
            # Shape: [num_frames, num_rx_ant]
            slow_time_matrix = y_frames[:, :, sym, sc] * phase_comp[sc]

            # Apply Hann window to reduce spectral leakage
            window = np.hanning(num_frames)[:, None]
            fft_matrix = np.fft.fft(slow_time_matrix * window, n=num_frames, axis=0)
            power_spectrum = np.abs(fft_matrix) ** 2  # [num_frames, num_rx_ant]

            # Sum over RX antennas → one spectrum per RE
            accumulated_spectrum += np.sum(power_spectrum, axis=1)

    # Find Doppler peak using fftshift for symmetric spectrum
    freqs = np.fft.fftshift(np.fft.fftfreq(num_frames, d=frame_interval_s))
    spectrum_shifted = np.fft.fftshift(accumulated_spectrum)

    peak_idx = int(np.argmax(spectrum_shifted))
    peak_val = spectrum_shifted[peak_idx]

    # Parabolic interpolation for sub-bin accuracy
    if 0 < peak_idx < num_frames - 1:
        alpha = spectrum_shifted[peak_idx - 1]
        beta = peak_val
        gamma = spectrum_shifted[peak_idx + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-12:
            delta = 0.5 * (alpha - gamma) / denom
        else:
            delta = 0.0
        doppler_hz = freqs[peak_idx] + delta * (freqs[1] - freqs[0])
    else:
        doppler_hz = freqs[peak_idx]

    wavelength = C / carrier_frequency_hz
    est_velocity_mps = doppler_hz * wavelength / 2.0

    return est_velocity_mps, doppler_hz


# ============================================================================
# Simulator
# ============================================================================


class ISACSimulator:
    """End-to-end ISAC simulator using Sionna RT."""

    def __init__(self, config: SimulationConfig, device: str | None = None):
        self.config = config

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.rng = np.random.default_rng(config.seed)

        # Initialize waveform generator
        wf_config = config.get_waveform_config()
        self.waveform_gen = SionnaWaveformGenerator(wf_config, device=device, seed=config.seed)

        # OFDM symbol duration (including CP)
        scs_hz = config.subcarrier_spacing_khz * 1e3
        # 5G NR: slot_duration = 1ms / 2^mu, mu = log2(SCS/15kHz)
        scs_hz = config.subcarrier_spacing_khz * 1e3
        mu = int(round(np.log2(config.subcarrier_spacing_khz / 15.0)))
        slot_duration_s = 1e-3 / (2 ** mu)
        self.ofdm_symbol_duration_s = slot_duration_s / config.num_pusch_symbols

        # Angle grid for DOA
        self.angle_grid = np.linspace(-70, 70, 281)

        # Initialize HyperDOA estimator (trained once, reused across trials)
        # NOTE: num_snapshots must match the actual number of time samples at test time.
        # In ISAC simulation we have one snapshot per valid OFDM resource element
        # (RE = resource element). With PUSCH DMRS, ~7% of the grid is valid.
        # We estimate the typical snapshot count from the waveform config.
        # This is used only for training-time dimension matching; the actual
        # snapshot count at test time varies per trial.
        self._hyperdoa: HyperDOAEstimator | None = None
        hyperdoa_training_ms: float | None = None
        if "hyperdoa" in config.algorithms:
            if HyperDOAEstimator is None or HyperDOAConfig is None:
                raise ImportError(
                    "HyperDOA is requested but not available. "
                    "Install dependencies: pip install torch-hd && cd doa/HYPERDOA && pip install -e ."
                )
            x_probe, _, valid_probe = self.waveform_gen.generate_waveform(batch_size=1)
            est_valid_re = int(np.count_nonzero(valid_probe))
            hypercfg = HyperDOAConfig(
                num_antennas=config.num_rx_ant,
                num_sources=1,
                num_snapshots=est_valid_re,
                n_dimensions=10000,
                min_angle_deg=-90.0,
                max_angle_deg=90.0,
                precision_deg=0.5,
                min_separation_deg=10.0,
                device=device,
                snr_db=10.0,
                training_samples=500,
            )
            self._hyperdoa = HyperDOAEstimator(hypercfg, seed=config.seed)
            t0 = time.perf_counter()
            self._hyperdoa.train()
            hyperdoa_training_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[HyperDOA] Training done in {hyperdoa_training_ms:.0f} ms "
                  f"(T={est_valid_re}, {hypercfg.training_samples} samples).")

    def estimate_doa(
        self,
        y: np.ndarray,
        x_tx: np.ndarray,
        valid: np.ndarray,
        f_k: np.ndarray,
        range_est_m: float,
        tx_angle_est_deg: float,
        method: str,
    ) -> float:
        """Estimate DOA using specified method.

        Args:
            y: Received signal [num_rx_ant, num_symbols, num_subcarriers]
            x_tx: Transmitted signal [num_tx_ant, num_symbols, num_subcarriers]
            valid: Valid RE mask
            f_k: Subcarrier frequencies
            range_est_m: Estimated range for phase compensation
            tx_angle_est_deg: Estimated TX angle for steering
            method: "music", "music_robust", or "hyperdoa"

        Returns:
            doa_est_deg: Estimated DOA in degrees
        """
        if method == "music_robust":
            raise NotImplementedError(
                "music_robust is not implemented yet. Use 'music' or 'hyperdoa'."
            )

        if method == "hyperdoa":
            # HyperDOA: use range-compensated snapshots as [N, T]
            # Range compensation removes the range-dependent phase from each subcarrier
            tau_hat = 2 * range_est_m / C
            h_range = np.exp(-1j * 2 * np.pi * f_k * tau_hat)

            a_tx = steering_vector_ula(self.config.num_tx_ant, tx_angle_est_deg)
            tx_field = np.einsum("t,tnk->nk", a_tx, x_tx)

            # Collect snapshots from valid REs
            snapshots = []
            for sym in range(valid.shape[0]):
                for sc in range(valid.shape[1]):
                    if valid[sym, sc] and abs(tx_field[sym, sc]) > 1e-8:
                        denom = tx_field[sym, sc] * h_range[sc]
                        snapshots.append(y[:, sym, sc] / denom)

            snapshots = np.asarray(snapshots, dtype=np.complex128)  # [num_snapshots, num_rx_ant]
            if snapshots.shape[0] < self.config.num_rx_ant:
                return float(self.angle_grid[0])

            # HyperDOA expects [N, T] — transpose [T, N] → [N, T]
            X_2d = snapshots.T  # [num_rx_ant, num_snapshots]
            # Scale to match HDC training signal level (~1-10 amplitude)
            power_scale = np.mean(np.abs(X_2d) ** 2)
            if power_scale > 0:
                X_2d = X_2d / np.sqrt(power_scale)
            return self._hyperdoa.predict_single(X_2d)

        # MUSIC-based methods
        return estimate_doa_music_after_range(
            y=y,
            x=x_tx,
            valid=valid,
            f_k=f_k,
            range_est_m=range_est_m,
            tx_angle_est_deg=tx_angle_est_deg,
            num_tx_ant=self.config.num_tx_ant,
            num_rx_ant=self.config.num_rx_ant,
            angle_grid_deg=self.angle_grid,
        )

    def run_single_trial(
        self,
        rt_config: SionnaRTConfig,
        snr_db: float,
        seed: int,
        doa_method: str = "music",
    ) -> tuple[float, float]:
        """Run one trial of the full simulation.

        Args:
            doa_method: DOA algorithm — "music", "music_robust", or "hyperdoa"

        Returns:
            range_est_m: Estimated range
            doa_est_deg: Estimated DOA
        """
        rng = np.random.default_rng(seed)

        # Generate waveform
        x_freq, _, valid_mask = self.waveform_gen.generate_waveform(batch_size=1)
        x_tx = x_freq[0, 0]  # [num_tx_ant, num_symbols, num_subcarriers]

        # Get subcarrier frequencies
        scs_hz = self.config.subcarrier_spacing_khz * 1e3
        k_centered = np.arange(x_tx.shape[-1]) - (x_tx.shape[-1] - 1) / 2
        f_k = k_centered * scs_hz

        # Initialize RT channel
        rt_channel = SionnaRTChannel(rt_config, device=self.device, seed=seed)

        # Velocity vector
        vel = (
            np.array(self.config.target_velocity_mps)
            if self.config.target_velocity_mps
            else None
        )

        # Simulate echo
        y_echo, _, _ = rt_channel.simulate_echo(
            x_tx=x_tx[None, :, :, :],  # Add batch dim: [1, num_tx_ant, num_symbols, num_subcarriers]
            valid_mask=valid_mask,
            subcarrier_spacing_hz=scs_hz,
            snr_db=snr_db,
            rng=rng,
            num_rx_ant=self.config.num_rx_ant,
            target_velocity_mps=vel,
            ofdm_symbol_duration_s=self.ofdm_symbol_duration_s,
        )

        y = y_echo[0]  # [num_rx_ant, num_symbols, num_subcarriers]

        # Coarse-to-fine range-angle estimation
        range_est, angle_est = coarse_to_fine_range_angle(
            y=y,
            x=x_tx,
            valid=valid_mask,
            f_k=f_k,
            num_tx_ant=self.config.num_tx_ant,
        )

        # DOA with selected method
        doa_est = self.estimate_doa(
            y=y,
            x_tx=x_tx,
            valid=valid_mask,
            f_k=f_k,
            range_est_m=range_est,
            tx_angle_est_deg=angle_est,
            method=doa_method,
        )

        return range_est, doa_est

    def run_single_observation(
        self,
        rt_config: SionnaRTConfig,
        snr_db: float,
        seed: int,
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate one shared ISAC observation for fair DOA-method comparison.

        Returns:
            range_est_m: Estimated range
            angle_est_deg: Coarse TX angle estimate
            y: Received signal [num_rx_ant, num_symbols, num_subcarriers]
            x_tx: Transmitted signal [num_tx_ant, num_symbols, num_subcarriers]
            valid_mask: Valid RE mask
            f_k: Subcarrier frequencies
        """
        rng = np.random.default_rng(seed)

        x_freq, _, valid_mask = self.waveform_gen.generate_waveform(batch_size=1)
        x_tx = x_freq[0, 0]

        scs_hz = self.config.subcarrier_spacing_khz * 1e3
        k_centered = np.arange(x_tx.shape[-1]) - (x_tx.shape[-1] - 1) / 2
        f_k = k_centered * scs_hz

        rt_channel = SionnaRTChannel(rt_config, device=self.device, seed=seed)

        vel = (
            np.array(self.config.target_velocity_mps)
            if self.config.target_velocity_mps
            else None
        )

        y_echo, _, _ = rt_channel.simulate_echo(
            x_tx=x_tx[None, :, :, :],
            valid_mask=valid_mask,
            subcarrier_spacing_hz=scs_hz,
            snr_db=snr_db,
            rng=rng,
            num_rx_ant=self.config.num_rx_ant,
            target_velocity_mps=vel,
            ofdm_symbol_duration_s=self.ofdm_symbol_duration_s,
        )

        y = y_echo[0]

        range_est, angle_est = coarse_to_fine_range_angle(
            y=y,
            x=x_tx,
            valid=valid_mask,
            f_k=f_k,
            num_tx_ant=self.config.num_tx_ant,
        )

        return range_est, angle_est, y, x_tx, valid_mask, f_k

    def run_experiment(
        self,
        model_type: ChannelModelType,
        snr_db: float,
    ) -> list[SimulationResult]:
        """Run Monte Carlo trials for one configuration.

        All DOA methods share the same per-trial observation (waveform, channel,
        noise, range estimate) — only the DOA estimator differs.  This ensures
        a fair algorithm comparison.

        Args:
            model_type: Channel model type
            snr_db: Sensing SNR in dB

        Returns:
            List of SimulationResult, one per configured DOA method.
        """
        rt_config = self.config.get_rt_config(model_type)

        uav_pos = np.array(self.config.uav_position)
        bs_pos = np.array(self.config.bs_position)
        true_range = float(np.linalg.norm(uav_pos - bs_pos))
        true_bearing = float(np.rad2deg(np.arctan2(uav_pos[1] - bs_pos[1], uav_pos[0] - bs_pos[0])))
        true_doa = -true_bearing  # Sionna convention

        # Per-method stats
        stats = {
            method: {"range_errors": [], "doa_errors": [], "runtime_s": 0.0}
            for method in self.config.algorithms
        }

        for trial in range(self.config.num_trials):
            seed = self.config.seed + trial * 100 + int(snr_db * 10)

            try:
                range_est, angle_est, y, x_tx, valid_mask, f_k = self.run_single_observation(
                    rt_config=rt_config, snr_db=snr_db, seed=seed
                )

                for method in self.config.algorithms:
                    t0 = time.perf_counter()

                    doa_est = self.estimate_doa(
                        y=y,
                        x_tx=x_tx,
                        valid=valid_mask,
                        f_k=f_k,
                        range_est_m=range_est,
                        tx_angle_est_deg=angle_est,
                        method=method,
                    )

                    stats[method]["runtime_s"] += time.perf_counter() - t0
                    stats[method]["range_errors"].append(range_est - true_range)
                    stats[method]["doa_errors"].append(doa_est - true_doa)

            except Exception as e:
                raise RuntimeError(f"Trial {trial} failed: {e}") from e

        results = []
        for method in self.config.algorithms:
            range_errors = np.array(stats[method]["range_errors"])
            doa_errors = np.array(stats[method]["doa_errors"])

            results.append(
                SimulationResult(
                    channel_type=model_type.value,
                    doa_method=method,
                    max_depth=rt_config.max_depth,
                    specular_reflection=rt_config.specular_reflection,
                    diffuse_scattering=rt_config.diffuse_scattering,
                    target_velocity_mps=(
                        float(np.linalg.norm(self.config.target_velocity_mps))
                        if self.config.target_velocity_mps
                        else 0.0
                    ),
                    snr_db=snr_db,
                    range_rmse_m=float(np.sqrt(np.mean(range_errors**2))),
                    range_bias_m=float(np.mean(range_errors)),
                    range_std_m=float(np.std(range_errors)),
                    doa_rmse_deg=float(np.sqrt(np.mean(doa_errors**2))),
                    doa_bias_deg=float(np.mean(doa_errors)),
                    doa_std_deg=float(np.std(doa_errors)),
                    runtime_ms=float(
                        stats[method]["runtime_s"] * 1000.0 / self.config.num_trials
                    ),
                    num_trials=self.config.num_trials,
                )
            )

        return results


# ============================================================================
# Main
# ============================================================================


def run_channel_comparison(config: SimulationConfig) -> list[SimulationResult]:
    """Run experiments comparing different channel models."""
    results = []

    channel_models = [
        (ChannelModelType.LOS, "LOS (直视)"),
        (ChannelModelType.NLOS, "NLoS (反射)"),
        (ChannelModelType.STREET_CANYON, "StreetCanyon (城市峡谷)"),
    ]

    snr_values = np.arange(config.snr_min_db, config.snr_max_db + 1e-9, config.snr_step_db)

    # Build simulator once — HyperDOA trains once and is reused across all channels/SNRs
    simulator = ISACSimulator(config)

    for model_type, model_name in channel_models:
        print(f"\n{'=' * 60}")
        print(f"Channel: {model_name}")
        print(f"{'=' * 60}")

        for snr_db in snr_values:
            print(f"  SNR = {snr_db:.1f} dB...", end=" ")

            result_list = simulator.run_experiment(model_type, snr_db)

            for res in result_list:
                print(
                    f"[{res.channel_type}] "
                    f"Range RMSE = {res.range_rmse_m:.2f} m, "
                    f"DOA RMSE = {res.doa_rmse_deg:.2f} deg"
                )
                results.append(res)

    return results


def run_multipath_study(config: SimulationConfig) -> list[SimulationResult]:
    """Study impact of multipath depth on performance."""
    results = []

    depth_configs = [
        (0, False, False, "LoS only"),
        (1, True, False, "LoS + specular, depth=1"),
        (2, True, False, "LoS + specular, depth=2"),
        (3, True, True, "Full multipath, depth=3"),
    ]

    snr_db = 10.0

    for max_depth, spec, diff, name in depth_configs:
        print(f"\n{'=' * 60}")
        print(f"Multipath config: {name}")
        print(f"{'=' * 60}")

        # Create temporary config with specific settings
        rt_config = config.get_rt_config(ChannelModelType.STREET_CANYON)
        rt_config.max_depth = max_depth
        rt_config.specular_reflection = spec
        rt_config.diffuse_scattering = diff

        range_errors = []
        doa_errors = []

        simulator = ISACSimulator(config)
        rng = np.random.default_rng(config.seed)

        start_time = time.perf_counter()

        for trial in range(config.num_trials):
            seed = config.seed + trial * 100

            try:
                x_freq, _, valid_mask = simulator.waveform_gen.generate_waveform(batch_size=1)
                x_tx = x_freq[0, 0]  # [num_tx_ant, num_symbols, num_subcarriers]
                scs_hz = config.subcarrier_spacing_khz * 1e3
                k_centered = np.arange(x_tx.shape[-1]) - (x_tx.shape[-1] - 1) / 2
                f_k = k_centered * scs_hz

                rt_channel = SionnaRTChannel(rt_config, device=simulator.device, seed=seed)

                vel = (
                    np.array(config.target_velocity_mps)
                    if config.target_velocity_mps
                    else None
                )

                y_echo, _, _ = rt_channel.simulate_echo(
                    x_tx=x_tx[None, :, :, :],
                    valid_mask=valid_mask,
                    subcarrier_spacing_hz=scs_hz,
                    snr_db=snr_db,
                    rng=rng,
                    num_rx_ant=config.num_rx_ant,
                    target_velocity_mps=vel,
                    ofdm_symbol_duration_s=simulator.ofdm_symbol_duration_s,
                )

                y = y_echo[0]

                range_est, angle_est = coarse_to_fine_range_angle(
                    y=y, x=x_tx, valid=valid_mask, f_k=f_k, num_tx_ant=config.num_tx_ant,
                )

                uav_pos = np.array(config.uav_position)
                bs_pos = np.array(config.bs_position)
                true_range = float(np.linalg.norm(uav_pos - bs_pos))
                true_bearing = float(np.rad2deg(np.arctan2(uav_pos[1] - bs_pos[1], uav_pos[0] - bs_pos[0])))
                true_doa = -true_bearing

                doa_est = simulator.estimate_doa(
                    y=y, x_tx=x_tx, valid=valid_mask, f_k=f_k,
                    range_est_m=range_est, tx_angle_est_deg=angle_est,
                    method="music",
                )

                range_errors.append(range_est - true_range)
                doa_errors.append(doa_est - true_doa)

            except Exception as e:
                raise RuntimeError(f"Trial {trial} failed: {e}") from e

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / config.num_trials

        range_errors = np.array(range_errors)
        doa_errors = np.array(doa_errors)

        results.append(
            SimulationResult(
                channel_type=name,
                doa_method="music",
                max_depth=max_depth,
                specular_reflection=spec,
                diffuse_scattering=diff,
                target_velocity_mps=(
                    float(np.linalg.norm(config.target_velocity_mps))
                    if config.target_velocity_mps
                    else 0.0
                ),
                snr_db=snr_db,
                range_rmse_m=float(np.sqrt(np.mean(range_errors**2))),
                range_bias_m=float(np.mean(range_errors)),
                range_std_m=float(np.std(range_errors)),
                doa_rmse_deg=float(np.sqrt(np.mean(doa_errors**2))),
                doa_bias_deg=float(np.mean(doa_errors)),
                doa_std_deg=float(np.std(doa_errors)),
                runtime_ms=elapsed_ms,
                num_trials=config.num_trials,
            )
        )

        print(f"  Range RMSE = {results[-1].range_rmse_m:.2f} m, DOA RMSE = {results[-1].doa_rmse_deg:.2f} deg")

    return results


def run_moving_target_study(config: SimulationConfig) -> list[SimulationResult]:
    """Study impact of target velocity on performance.

    Uses multi-frame slow-time observation with range-Doppler velocity estimation.
    """
    results = []

    velocity_configs = [
        ((0.0, 0.0, 0.0), "Static"),
        ((10.0, 0.0, 0.0), "10 m/s (rx)"),
        ((20.0, 0.0, 0.0), "20 m/s (rx)"),
        ((-10.0, 0.0, 0.0), "10 m/s (tx)"),
    ]

    snr_db = 10.0
    num_frames = 128          # More frames → finer Doppler bin spacing
    frame_interval_s = 5e-4   # 0.5 ms between frames (2000 Hz PRF, covers ±42 m/s)

    for velocity, name in velocity_configs:
        print(f"\n{'=' * 60}")
        print(f"Target velocity: {name}")
        print(f"{'=' * 60}")

        config_copy = SimulationConfig(
            mode=config.mode,
            snr_min_db=config.snr_min_db,
            snr_max_db=config.snr_max_db,
            snr_step_db=config.snr_step_db,
            num_trials=config.num_trials,
            num_tx_ant=config.num_tx_ant,
            num_rx_ant=config.num_rx_ant,
            num_pusch_symbols=config.num_pusch_symbols,
            subcarrier_spacing_khz=config.subcarrier_spacing_khz,
            num_rbs=config.num_rbs,
            bs_position=config.bs_position,
            uav_position=config.uav_position,
            target_velocity_mps=velocity,
            algorithms=config.algorithms,
            output_dir=config.output_dir,
            seed=config.seed,
        )

        rt_config = config_copy.get_rt_config(ChannelModelType.STREET_CANYON)
        rt_config.max_depth = 1  # Keep multipath moderate for velocity study
        rt_config.specular_reflection = True
        rt_config.diffuse_scattering = False

        range_errors = []
        doa_errors = []
        vel_errors = []

        simulator = ISACSimulator(config_copy)

        start_time = time.perf_counter()

        for trial in range(config_copy.num_trials):
            seed = config_copy.seed + trial * 100

            try:
                x_freq, _, valid_mask = simulator.waveform_gen.generate_waveform(batch_size=1)
                x_tx = x_freq[0, 0]  # [num_tx_ant, num_symbols, num_subcarriers]
                scs_hz = config_copy.subcarrier_spacing_khz * 1e3
                k_centered = np.arange(x_tx.shape[-1]) - (x_tx.shape[-1] - 1) / 2
                f_k = k_centered * scs_hz

                rt_channel = SionnaRTChannel(rt_config, device=simulator.device, seed=seed)
                rng = np.random.default_rng(seed)

                vel = np.array(velocity)

                # Multi-frame echo with moving target position
                y_frames = rt_channel.simulate_echo_multi_frame(
                    x_tx=x_tx,
                    valid_mask=valid_mask,
                    subcarrier_spacing_hz=scs_hz,
                    snr_db=snr_db,
                    num_frames=num_frames,
                    frame_interval_s=frame_interval_s,
                    initial_uav_pos=np.array(config_copy.uav_position),
                    target_velocity_mps=vel,
                    bs_pos=np.array(config_copy.bs_position),
                    rng=rng,
                    num_rx_ant=config_copy.num_rx_ant,
                    update_channel_every=num_frames,  # Fixed geometry for full CPI (coherent processing interval)
                )

                # Range and DOA estimation from first frame
                y = y_frames[0]  # [num_rx_ant, num_symbols, num_subcarriers]

                range_est, angle_est = coarse_to_fine_range_angle(
                    y=y, x=x_tx, valid=valid_mask, f_k=f_k, num_tx_ant=config_copy.num_tx_ant,
                )

                uav_pos = np.array(config_copy.uav_position)
                bs_pos = np.array(config_copy.bs_position)
                true_range = float(np.linalg.norm(uav_pos - bs_pos))
                true_bearing = float(np.rad2deg(np.arctan2(uav_pos[1] - bs_pos[1], uav_pos[0] - bs_pos[0])))
                true_doa = -true_bearing

                doa_est = simulator.estimate_doa(
                    y=y, x_tx=x_tx, valid=valid_mask, f_k=f_k,
                    range_est_m=range_est, tx_angle_est_deg=angle_est,
                    method="music",
                )

                # True radial velocity: dot(v, unit_LOS)
                rel = uav_pos - bs_pos
                distance = np.linalg.norm(rel) + 1e-12
                unit_los = rel / distance
                true_radial_vel = float(np.dot(vel, unit_los))

                # Radial velocity estimation via slow-time Doppler FFT
                vel_est, doppler_hz = estimate_radial_velocity_doppler(
                    y_frames=y_frames,
                    valid_mask=valid_mask,
                    range_est_m=range_est,
                    f_k=f_k,
                    frame_interval_s=frame_interval_s,
                    carrier_frequency_hz=3.5e9,
                )

                range_errors.append(range_est - true_range)
                doa_errors.append(doa_est - true_doa)
                vel_errors.append(vel_est - true_radial_vel)

            except Exception as e:
                raise RuntimeError(f"Trial {trial} failed: {e}") from e

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / config_copy.num_trials

        range_errors = np.array(range_errors)
        doa_errors = np.array(doa_errors)
        vel_errors = np.array(vel_errors)

        results.append(
            SimulationResult(
                channel_type=name,
                doa_method="music",
                max_depth=rt_config.max_depth,
                specular_reflection=rt_config.specular_reflection,
                diffuse_scattering=rt_config.diffuse_scattering,
                target_velocity_mps=float(np.linalg.norm(velocity)),
                snr_db=snr_db,
                range_rmse_m=float(np.sqrt(np.mean(range_errors**2))),
                range_bias_m=float(np.mean(range_errors)),
                range_std_m=float(np.std(range_errors)),
                doa_rmse_deg=float(np.sqrt(np.mean(doa_errors**2))),
                doa_bias_deg=float(np.mean(doa_errors)),
                doa_std_deg=float(np.std(doa_errors)),
                vel_rmse_mps=float(np.sqrt(np.mean(vel_errors**2))),
                vel_bias_mps=float(np.mean(vel_errors)),
                vel_std_mps=float(np.std(vel_errors)),
                true_radial_velocity_mps=np.dot(np.array(velocity), (np.array(config_copy.uav_position) - np.array(config_copy.bs_position)) / (np.linalg.norm(np.array(config_copy.uav_position) - np.array(config_copy.bs_position)) + 1e-12)),
                runtime_ms=elapsed_ms,
                num_trials=config_copy.num_trials,
            )
        )

        v_rmse = results[-1].vel_rmse_mps
        print(f"  Range RMSE = {results[-1].range_rmse_m:.2f} m, "
              f"DOA RMSE = {results[-1].doa_rmse_deg:.2f} deg, "
              f"Vel RMSE = {v_rmse:.2f} m/s")

    return results


def plot_results(results: list[SimulationResult], output_dir: Path) -> None:
    """Plot simulation results."""
    channel_types = sorted(set(r.channel_type for r in results))
    doa_methods = sorted(set(r.doa_method for r in results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ch_colors = {"LOS": "#3182bd", "NLoS": "#31a354", "StreetCanyon": "#e6550d"}
    ch_markers = {"LOS": "o", "NLoS": "s", "StreetCanyon": "^"}
    method_linestyles = {"music": "-", "music_robust": "--", "hyperdoa": ":", "esprit": "-.",}

    for ch_type in channel_types:
        for method in doa_methods:
            ch_results = [r for r in results if r.channel_type == ch_type and r.doa_method == method]
            if not ch_results:
                continue
            snrs = [r.snr_db for r in ch_results]
            rmses = [r.doa_rmse_deg for r in ch_results]
            label = f"{ch_type} [{method}]"
            ls = method_linestyles.get(method, "-")
            axes[0].plot(snrs, rmses, marker=ch_markers.get(ch_type, "o"),
                         color=ch_colors.get(ch_type, "gray"), label=label,
                         linewidth=1.5, markersize=7, linestyle=ls)

    axes[0].set_xlabel("SNR (dB)", fontsize=12)
    axes[0].set_ylabel("DOA RMSE (deg)", fontsize=12)
    axes[0].set_title("DOA Estimation vs SNR", fontsize=14)
    axes[0].legend(fontsize=8, loc="best")
    axes[0].grid(True, alpha=0.3)

    for ch_type in channel_types:
        for method in doa_methods:
            ch_results = [r for r in results if r.channel_type == ch_type and r.doa_method == method]
            if not ch_results:
                continue
            snrs = [r.snr_db for r in ch_results]
            rmses = [r.range_rmse_m for r in ch_results]
            label = f"{ch_type} [{method}]"
            ls = method_linestyles.get(method, "-")
            axes[1].plot(snrs, rmses, marker=ch_markers.get(ch_type, "o"),
                         color=ch_colors.get(ch_type, "gray"), label=label,
                         linewidth=1.5, markersize=7, linestyle=ls)

    axes[1].set_xlabel("SNR (dB)", fontsize=12)
    axes[1].set_ylabel("Range RMSE (m)", fontsize=12)
    axes[1].set_title("Range Estimation vs SNR", fontsize=14)
    axes[1].legend(fontsize=8, loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "end2end_rmse_vs_snr.png", dpi=180)
    plt.close(fig)


def save_results(results: list[SimulationResult], output_dir: Path) -> None:
    """Save results to CSV."""
    output_path = output_dir / "simulation_results.csv"

    fieldnames = [
        "channel_type", "doa_method", "max_depth",
        "specular_reflection", "diffuse_scattering",
        "target_velocity_mps", "true_radial_velocity_mps", "snr_db",
        "range_rmse_m", "range_bias_m", "range_std_m",
        "doa_rmse_deg", "doa_bias_deg", "doa_std_deg",
        "vel_rmse_mps", "vel_bias_mps", "vel_std_mps",
        "runtime_ms", "num_trials",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "channel_type": result.channel_type,
                "doa_method": result.doa_method,
                "max_depth": result.max_depth,
                "specular_reflection": result.specular_reflection,
                "diffuse_scattering": result.diffuse_scattering,
                "target_velocity_mps": f"{result.target_velocity_mps:.2f}",
                "snr_db": f"{result.snr_db:.1f}",
                "range_rmse_m": f"{result.range_rmse_m:.4f}",
                "range_bias_m": f"{result.range_bias_m:.4f}",
                "range_std_m": f"{result.range_std_m:.4f}",
                "doa_rmse_deg": f"{result.doa_rmse_deg:.4f}",
                "doa_bias_deg": f"{result.doa_bias_deg:.4f}",
                "doa_std_deg": f"{result.doa_std_deg:.4f}",
                "runtime_ms": f"{result.runtime_ms:.2f}",
                "num_trials": result.num_trials,
            }
            if result.vel_rmse_mps is not None:
                row["vel_rmse_mps"] = f"{result.vel_rmse_mps:.4f}"
                row["vel_bias_mps"] = f"{result.vel_bias_mps:.4f}"
                row["vel_std_mps"] = f"{result.vel_std_mps:.4f}"
                row["true_radial_velocity_mps"] = f"{result.true_radial_velocity_mps:.4f}" if result.true_radial_velocity_mps is not None else ""
            else:
                row["vel_rmse_mps"] = ""
                row["vel_bias_mps"] = ""
                row["vel_std_mps"] = ""
                row["true_radial_velocity_mps"] = ""
            writer.writerow(row)

    print(f"\nResults saved to: {output_path}")


def print_results_table(results: list[SimulationResult]) -> None:
    """Print a formatted table of results."""
    print("\n" + "=" * 155)
    print(f"{'Channel':<22} {'DOA':<10} {'MaxD':>5} {'Spec':>5} {'Diff':>5} {'Vel':>6} "
          f"{'TrueRadial':>10} {'SNR':>6} {'RangeRMSE':>10} {'DOARMSE':>10} {'VelRMSE':>10} {'VelStd':>8}")
    print("-" * 155)

    for r in results:
        vel_str = f"{r.vel_rmse_mps:.2f}" if r.vel_rmse_mps is not None else "      N/A"
        vel_std_str = f"{r.vel_std_mps:.2f}" if r.vel_std_mps is not None else "     N/A"
        true_rad_str = f"{r.true_radial_velocity_mps:.2f}" if r.true_radial_velocity_mps is not None else "        N/A"
        print(
            f"{r.channel_type:<22} {r.doa_method:<10} {r.max_depth:>5} "
            f"{str(r.specular_reflection):>5} {str(r.diffuse_scattering):>5} "
            f"{r.target_velocity_mps:>6.1f} {true_rad_str:>10} {r.snr_db:>6.1f} "
            f"{r.range_rmse_m:>10.3f} {r.doa_rmse_deg:>10.3f} {vel_str:>10} {vel_std_str:>8}"
        )

    print("=" * 155)


def main() -> None:
    parser = argparse.ArgumentParser(description="ISAC End-to-End Simulation with Sionna RT")

    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "los", "nlos", "quick"])
    parser.add_argument("--snr-min", type=float, default=-5.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--snr-step", type=float, default=5.0)
    parser.add_argument("--num-trials", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-channels", action="store_true")
    parser.add_argument("--multipath-study", action="store_true")
    parser.add_argument("--moving-target-study", action="store_true")
    parser.add_argument(
        "--doa-methods",
        nargs="+",
        default=["music"],
        choices=["music", "hyperdoa"],
        help="DOA estimators to evaluate (default: music)",
    )

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = SimulationConfig(
        mode=args.mode,
        snr_min_db=args.snr_min,
        snr_max_db=args.snr_max,
        snr_step_db=args.snr_step,
        num_trials=args.num_trials,
        seed=args.seed,
        algorithms=tuple(args.doa_methods),
    )

    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    print(f"\nISAC End-to-End Simulation")
    print(f"Device: {device}")
    print(f"Target position: {config.uav_position} m")
    print(f"Output: {config.output_dir}")

    all_results = []

    if args.compare_channels:
        results = run_channel_comparison(config)
        all_results.extend(results)
        plot_results(results, config.output_dir)

    if args.multipath_study:
        results = run_multipath_study(config)
        all_results.extend(results)

    if args.moving_target_study:
        results = run_moving_target_study(config)
        all_results.extend(results)

    if not args.compare_channels and not args.multipath_study and not args.moving_target_study:
        # Default: run channel comparison
        results = run_channel_comparison(config)
        all_results.extend(results)
        plot_results(results, config.output_dir)

    if all_results:
        save_results(all_results, config.output_dir)
        print_results_table(all_results)

        print(f"\n{'=' * 60}")
        print(f"Simulation Complete")
        print(f"Total results: {len(all_results)}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()