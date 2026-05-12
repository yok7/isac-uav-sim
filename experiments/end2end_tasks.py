"""Experiment task functions: channel comparison, multipath study, moving target study."""

from __future__ import annotations

import time
import numpy as np

from channel import ChannelModelType, SionnaRTChannel
from experiments.end2end_config import SimulationConfig, SimulationResult
from experiments.end2end_simulator import ISACSimulator
from sensing.range_estimation import coarse_to_fine_range_angle
from sensing.velocity_estimation import estimate_radial_velocity_doppler


# ---------------------------------------------------------------------------
# Channel comparison
# ---------------------------------------------------------------------------

def run_channel_comparison(config: SimulationConfig) -> list[SimulationResult]:
    """Run experiments comparing different channel models.

    All three channel types (LOS, NLoS, StreetCanyon) share one ISACSimulator
    instance so that HyperDOA is trained once and reused across all SNR points.

    Args:
        config: Simulation configuration

    Returns:
        List of SimulationResult, one per (channel, SNR, DOA method) combination.
    """
    results = []

    channel_models = [
        (ChannelModelType.LOS, "LOS (直视)"),
        (ChannelModelType.NLOS, "NLoS (反射)"),
        (ChannelModelType.STREET_CANYON, "StreetCanyon (城市峡谷)"),
    ]

    snr_values = np.arange(
        config.snr_min_db,
        config.snr_max_db + 1e-9,
        config.snr_step_db,
    )

    # Build simulator once — HyperDOA trains once and is reused
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
                    f"[{res.channel_type} | {res.doa_method}] "
                    f"Range RMSE = {res.range_rmse_m:.2f} m, "
                    f"DOA RMSE = {res.doa_rmse_deg:.2f} deg"
                )
                results.append(res)

    return results


# ---------------------------------------------------------------------------
# Multipath depth study
# ---------------------------------------------------------------------------

def run_multipath_study(config: SimulationConfig) -> list[SimulationResult]:
    """Study impact of multipath depth on performance.

    Varies max_depth, specular_reflection, diffuse_scattering while keeping
    all other parameters fixed (StreetCanyon geometry, SNR=10 dB).

    Args:
        config: Simulation configuration

    Returns:
        List of SimulationResult, one per multipath configuration.
    """
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
                true_bearing = float(
                    np.rad2deg(
                        np.arctan2(
                            uav_pos[1] - bs_pos[1],
                            uav_pos[0] - bs_pos[0],
                        )
                    )
                )
                true_doa = -true_bearing

                doa_est = simulator.doa_bank.estimate(
                    method="music",
                    y=y, x_tx=x_tx, valid=valid_mask, f_k=f_k,
                    range_est_m=range_est, tx_angle_est_deg=angle_est,
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

        print(
            f"  Range RMSE = {results[-1].range_rmse_m:.2f} m, "
            f"DOA RMSE = {results[-1].doa_rmse_deg:.2f} deg"
        )

    return results


# ---------------------------------------------------------------------------
# Moving target / velocity study
# ---------------------------------------------------------------------------

def run_moving_target_study(config: SimulationConfig) -> list[SimulationResult]:
    """Study impact of target velocity on performance.

    Uses multi-frame slow-time observation with range-Doppler velocity estimation.
    Each velocity configuration gets its own ISACSimulator so that DOA method
    selection is per-configuration.

    Args:
        config: Simulation configuration

    Returns:
        List of SimulationResult, one per velocity configuration.
    """
    results = []

    velocity_configs = [
        ((0.0, 0.0, 0.0), "Static"),
        ((10.0, 0.0, 0.0), "10 m/s (rx)"),
        ((20.0, 0.0, 0.0), "20 m/s (rx)"),
        ((-10.0, 0.0, 0.0), "10 m/s (tx)"),
    ]

    snr_db = 10.0
    num_frames = 128           # More frames → finer Doppler bin spacing
    frame_interval_s = 5e-4    # 0.5 ms between frames (2000 Hz PRF, covers ±42 m/s)

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
        rt_config.max_depth = 1  # Moderate multipath for velocity study
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
                    update_channel_every=num_frames,  # Fixed geometry for full CPI
                )

                # Range and DOA estimation from first frame
                y = y_frames[0]  # [num_rx_ant, num_symbols, num_subcarriers]

                range_est, angle_est = coarse_to_fine_range_angle(
                    y=y, x=x_tx, valid=valid_mask, f_k=f_k,
                    num_tx_ant=config_copy.num_tx_ant,
                )

                uav_pos = np.array(config_copy.uav_position)
                bs_pos = np.array(config_copy.bs_position)
                true_range = float(np.linalg.norm(uav_pos - bs_pos))
                true_bearing = float(
                    np.rad2deg(
                        np.arctan2(
                            uav_pos[1] - bs_pos[1],
                            uav_pos[0] - bs_pos[0],
                        )
                    )
                )
                true_doa = -true_bearing

                doa_est = simulator.doa_bank.estimate(
                    method="music",
                    y=y, x_tx=x_tx, valid=valid_mask, f_k=f_k,
                    range_est_m=range_est, tx_angle_est_deg=angle_est,
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
                true_radial_velocity_mps=np.dot(
                    np.array(velocity),
                    (
                        np.array(config_copy.uav_position)
                        - np.array(config_copy.bs_position)
                    ) / (
                        np.linalg.norm(
                            np.array(config_copy.uav_position)
                            - np.array(config_copy.bs_position)
                        ) + 1e-12
                    ),
                ),
                runtime_ms=elapsed_ms,
                num_trials=config_copy.num_trials,
            )
        )

        v_rmse = results[-1].vel_rmse_mps
        print(
            f"  Range RMSE = {results[-1].range_rmse_m:.2f} m, "
            f"DOA RMSE = {results[-1].doa_rmse_deg:.2f} deg, "
            f"Vel RMSE = {v_rmse:.2f} m/s"
        )

    return results