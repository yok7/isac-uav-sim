"""End-to-end ISAC simulator using Sionna RT.

Provides the core simulation loop that:
- Generates OFDM waveforms via Sionna PUSCH
- Simulates two-leg RT echo (BS→UAV→BS)
- Estimates range via coarse-to-fine search
- Estimates DOA via music or hyperdoa
- Returns per-method statistics
"""

from __future__ import annotations

import time
import numpy as np
import torch

from channel import ChannelModelType, SionnaRTChannel, SionnaRTConfig
from waveform import SionnaWaveformGenerator
from experiments.end2end_config import SimulationConfig, SimulationResult
from sensing.range_estimation import coarse_to_fine_range_angle
from doa.isac_doa import DOAEstimatorBank


class ISACSimulator:
    """End-to-end ISAC simulator using Sionna RT.

    Orchestrates the full sensing pipeline:
    1. Generate OFDM waveform
    2. Simulate RT echo (BS→UAV→BS)
    3. Estimate range and angle (coarse-to-fine)
    4. Estimate DOA (music or hyperdoa)
    5. Accumulate statistics per DOA method

    Each trial generates a shared observation (waveform, channel, noise,
    range estimate) that is then evaluated by all configured DOA methods.
    This ensures fair algorithm comparison.

    Example:
        >>> config = SimulationConfig(num_trials=30, algorithms=("music", "hyperdoa"))
        >>> sim = ISACSimulator(config)
        >>> results = sim.run_experiment(ChannelModelType.STREET_CANYON, snr_db=10.0)
        >>> for r in results:
        ...     print(f"{r.doa_method}: DOA RMSE={r.doa_rmse_deg:.2f} deg")
    """

    def __init__(self, config: SimulationConfig, device: str | None = None):
        """Initialize the simulator.

        Args:
            config: Simulation configuration
            device: "cuda" or "cpu". Defaults to cuda if available.
        """
        self.config = config

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.rng = np.random.default_rng(config.seed)

        # Waveform generator (initialized once, reused across all trials)
        wf_config = config.get_waveform_config()
        self.waveform_gen = SionnaWaveformGenerator(
            wf_config,
            device=device,
            seed=config.seed,
        )

        # OFDM symbol duration (including CP)
        scs_hz = config.subcarrier_spacing_khz * 1e3
        mu = int(round(np.log2(config.subcarrier_spacing_khz / 15.0)))
        slot_duration_s = 1e-3 / (2 ** mu)
        self.ofdm_symbol_duration_s = slot_duration_s / config.num_pusch_symbols

        # Angle grid for MUSIC
        self.angle_grid = np.linspace(-70, 70, 281)

        # Initialize DOA estimator bank (MUSIC + HyperDOA trained once)
        x_probe, _, valid_probe = self.waveform_gen.generate_waveform(batch_size=1)
        num_snapshots = int(np.count_nonzero(valid_probe))

        self.doa_bank = DOAEstimatorBank(
            algorithms=config.algorithms,
            num_rx_ant=config.num_rx_ant,
            num_tx_ant=config.num_tx_ant,
            num_snapshots=num_snapshots,
            device=device,
            seed=config.seed,
            angle_grid_deg=self.angle_grid,
        )

    def run_single_observation(
        self,
        rt_config: SionnaRTConfig,
        snr_db: float,
        seed: int,
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate one shared ISAC observation for fair DOA-method comparison.

        This is the core observation generator. It creates a single
        waveform + RT channel realization that is shared across all DOA
        estimators in the same trial. This eliminates the bias from
        different random draws.

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
        true_bearing = float(
            np.rad2deg(
                np.arctan2(
                    uav_pos[1] - bs_pos[1],
                    uav_pos[0] - bs_pos[0],
                )
            )
        )
        true_doa = -true_bearing  # Sionna convention: negative bearing = DOA

        # Per-method stats accumulators
        stats = {
            method: {"range_errors": [], "doa_errors": [], "runtime_s": 0.0}
            for method in self.config.algorithms
        }

        for trial in range(self.config.num_trials):
            seed = self.config.seed + trial * 100 + int(snr_db * 10)

            try:
                range_est, angle_est, y, x_tx, valid_mask, f_k = self.run_single_observation(
                    rt_config=rt_config,
                    snr_db=snr_db,
                    seed=seed,
                )

                for method in self.config.algorithms:
                    t0 = time.perf_counter()

                    doa_est = self.doa_bank.estimate(
                        method=method,
                        y=y,
                        x_tx=x_tx,
                        valid=valid_mask,
                        f_k=f_k,
                        range_est_m=range_est,
                        tx_angle_est_deg=angle_est,
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