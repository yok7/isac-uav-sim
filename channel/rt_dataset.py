"""Dataset generation utilities for Sionna RT channel responses.

This module generates CFR datasets from the split Sionna RT channel stack.

Design note:
    Do not import SionnaRTChannel from channel.sionna_rt_channel here,
    otherwise channel.sionna_rt_channel -> rt_dataset -> sionna_rt_channel
    can create a circular import. Instead, compose a local internal channel
    class from SionnaRTChannelCore and RTEchoMixin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from channel.rt_config import SionnaRTConfig
from channel.rt_core import SionnaRTChannelCore
from channel.rt_echo import RTEchoMixin


class _DatasetRTChannel(RTEchoMixin, SionnaRTChannelCore):
    """Internal RT channel class used only for dataset generation."""

    pass


def generate_cfr_dataset(
    config: SionnaRTConfig,
    num_cfrs: int,
    batch_size: int = 100,
    device: str | None = None,
    seed: int | None = 42,
    num_subcarriers: int = 128,
    subcarrier_spacing_hz: float = 30e3,
    position_jitter_m: tuple[float, float, float] = (5.0, 5.0, 2.0),
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a dataset of one-leg Sionna RT channel frequency responses.

    The generated CFR corresponds to the BS -> UAV propagation leg. The UAV
    position is randomly jittered around the nominal target position for each
    sample. This is useful for learning-based models or sanity checks that need
    multiple channel realizations around the same scenario.

    Args:
        config:
            Sionna RT configuration.
        num_cfrs:
            Number of CFR samples to generate.
        batch_size:
            Progress-printing batch size. The current implementation still
            generates one CFR at a time because each sample may use a different
            UAV position.
        device:
            Torch/Sionna device. If None, use CUDA when available.
        seed:
            Random seed for reproducible UAV position jitter.
        num_subcarriers:
            Number of subcarriers in the generated CFR.
        subcarrier_spacing_hz:
            Subcarrier spacing in Hz.
        position_jitter_m:
            Maximum absolute random jitter in x, y, z directions.
            For example, (5, 5, 2) means:
            x += U[-5, 5], y += U[-5, 5], z += U[-2, 2].
        save_path:
            Optional .npz path. If provided, the generated dataset is saved.

    Returns:
        A dictionary with:
            cfr:
                Complex CFR array with shape
                [num_cfrs, num_rx_ant, num_tx_ant, num_subcarriers].
                For the default BS->UAV leg this is
                [num_cfrs, 1, config.num_bs_ant, num_subcarriers].
            tx_positions:
                Array [num_cfrs, 3].
            rx_positions:
                Array [num_cfrs, 3].
            config:
                Lightweight metadata dictionary.
    """
    if num_cfrs <= 0:
        raise ValueError(f"num_cfrs must be positive, got {num_cfrs}")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(seed)

    channel = _DatasetRTChannel(config=config, device=device, seed=seed)

    cfr_list: list[np.ndarray] = []
    tx_positions: list[np.ndarray] = []
    rx_positions: list[np.ndarray] = []

    bs_pos = channel._bs_pos.copy()
    uav_pos_nominal = channel._uav_pos.copy()
    jitter_limits = np.asarray(position_jitter_m, dtype=float)

    num_batches = int(np.ceil(num_cfrs / batch_size))

    for sample_idx in range(num_cfrs):
        batch_idx = sample_idx // batch_size
        if sample_idx % batch_size == 0:
            print(
                f"Generating CFR batch {batch_idx + 1}/{num_batches} "
                f"({sample_idx + 1}/{num_cfrs})...",
                end="\r",
            )

        jitter = rng.uniform(-jitter_limits, jitter_limits)
        rx_pos = uav_pos_nominal + jitter

        h = channel.compute_leg_cfr(
            tx_position=bs_pos,
            rx_position=rx_pos,
            num_tx_ant=config.num_bs_ant,
            num_rx_ant=1,
            num_subcarriers=num_subcarriers,
            subcarrier_spacing_hz=subcarrier_spacing_hz,
            tx_name="bs_tx",
            rx_name="uav_probe",
            tx_look_at_rx=False,
            rx_look_at_tx=True,
            seed=(seed + sample_idx) if seed is not None else None,
        )

        cfr_list.append(h)
        tx_positions.append(bs_pos.copy())
        rx_positions.append(rx_pos.copy())

    print(f"\nGenerated {num_cfrs} CFR samples.")

    cfr = np.stack(cfr_list, axis=0)
    tx_positions_arr = np.stack(tx_positions, axis=0)
    rx_positions_arr = np.stack(rx_positions, axis=0)

    dataset = {
        "cfr": cfr,
        "tx_positions": tx_positions_arr,
        "rx_positions": rx_positions_arr,
        "config": {
            "model_type": config.model_type.value,
            "carrier_frequency_hz": config.carrier_frequency_hz,
            "max_depth": config.max_depth,
            "samples_per_src": config.samples_per_src,
            "specular_reflection": config.specular_reflection,
            "diffuse_scattering": config.diffuse_scattering,
            "num_bs_ant": config.num_bs_ant,
            "num_subcarriers": num_subcarriers,
            "subcarrier_spacing_hz": subcarrier_spacing_hz,
            "bs_position": list(config.bs_position),
            "uav_nominal_position": uav_pos_nominal.tolist(),
            "position_jitter_m": list(position_jitter_m),
        },
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            cfr=dataset["cfr"],
            tx_positions=dataset["tx_positions"],
            rx_positions=dataset["rx_positions"],
            config=np.array([dataset["config"]], dtype=object),
        )
        print(f"CFR dataset saved to: {save_path}")

    return dataset


def generate_cir_dataset(
    *args,
    **kwargs,
) -> dict[str, Any]:
    """Backward-compatible alias for old code.

    The current implementation generates CFR, not CIR. This alias is kept so
    old scripts that still call generate_cir_dataset() do not break immediately.
    Prefer using generate_cfr_dataset() in new code.
    """
    print(
        "[WARNING] generate_cir_dataset() is deprecated and now returns CFR data. "
        "Use generate_cfr_dataset() instead."
    )
    return generate_cfr_dataset(*args, **kwargs)