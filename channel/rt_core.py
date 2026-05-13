# channel/rt_core.py
from __future__ import annotations

import numpy as np
import torch

from sionna.rt import (
    load_scene,
    PathSolver,
    PlanarArray,
    Transmitter,
    Receiver,
    subcarrier_frequencies,
)

from channel.rt_config import (
    C,
    ChannelModelType,
    SionnaRTConfig,
    get_scene_path,
)


class SionnaRTChannelCore:
    """Core Sionna RT channel: scene setup and one-leg CFR computation."""

    def __init__(
        self,
        config: SionnaRTConfig,
        device: str | None = None,
        seed: int | None = 42,
    ):
        self.config = config

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.seed = seed
        self._path_solver = PathSolver()

        self._bs_pos = np.array(config.bs_position, dtype=float)
        self._uav_pos = config.get_uav_position()

    def _get_solver_options(self) -> dict:
        model_type = self.config.model_type

        if model_type == ChannelModelType.LOS:
            return {
                "los": True,
                "specular_reflection": False,
                "diffuse_reflection": False,
                "refraction": False,
                "diffraction": False,
            }

        if model_type in [ChannelModelType.NLOS, ChannelModelType.MIXED]:
            return {
                "los": model_type == ChannelModelType.MIXED,
                "specular_reflection": self.config.specular_reflection,
                "diffuse_reflection": self.config.diffuse_scattering,
                "refraction": False,
                "diffraction": False,
            }

        if model_type in [ChannelModelType.MULTIPATH, ChannelModelType.STREET_CANYON]:
            return {
                "los": True,
                "specular_reflection": self.config.specular_reflection,
                "diffuse_reflection": self.config.diffuse_scattering,
                "refraction": False,
                "diffraction": False,
            }

        return {
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": self.config.diffuse_scattering,
            "refraction": False,
            "diffraction": False,
        }

    @staticmethod
    def _make_ula_array(num_ant: int) -> PlanarArray:
        return PlanarArray(
            num_rows=1,
            num_cols=num_ant,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )

    def compute_leg_cfr(
        self,
        tx_position: np.ndarray,
        rx_position: np.ndarray,
        num_tx_ant: int,
        num_rx_ant: int,
        num_subcarriers: int,
        subcarrier_spacing_hz: float,
        tx_name: str = "tx",
        rx_name: str = "rx",
        tx_look_at_rx: bool = False,
        rx_look_at_tx: bool = False,
        seed: int | None = None,
    ) -> np.ndarray:
        """Compute one RT propagation leg.

        Returns:
            CFR with shape [num_rx_ant, num_tx_ant, num_subcarriers].
        """
        scene = load_scene(get_scene_path(self.config))
        scene.frequency = self.config.carrier_frequency_hz

        scene.tx_array = self._make_ula_array(num_tx_ant)
        scene.rx_array = self._make_ula_array(num_rx_ant)

        tx_kwargs = (
            {"look_at": rx_position.tolist()}
            if tx_look_at_rx
            else {"orientation": [0.0, 0.0, 0.0]}
        )
        rx_kwargs = (
            {"look_at": tx_position.tolist()}
            if rx_look_at_tx
            else {"orientation": [0.0, 0.0, 0.0]}
        )

        if tx_name in scene.transmitters:
            scene.remove(tx_name)
        if rx_name in scene.receivers:
            scene.remove(rx_name)

        scene.add(
            Transmitter(
                name=tx_name,
                position=tx_position.tolist(),
                power_dbm=30.0,
                **tx_kwargs,
            )
        )
        scene.add(
            Receiver(
                name=rx_name,
                position=rx_position.tolist(),
                **rx_kwargs,
            )
        )

        paths = self._path_solver(
            scene,
            max_depth=self.config.max_depth,
            samples_per_src=self.config.samples_per_src,
            synthetic_array=False,
            seed=seed if seed is not None else self.seed,
            **self._get_solver_options(),
        )

        freqs = subcarrier_frequencies(num_subcarriers, subcarrier_spacing_hz)

        h = paths.cfr(
            freqs,
            normalize=False,
            normalize_delays=False,
            out_type="numpy",
        )

        return np.asarray(h[0, :, 0, :, 0, :], dtype=np.complex128)

    def get_channel_stats(self) -> dict:
        h_out = self.compute_leg_cfr(
            tx_position=self._bs_pos,
            rx_position=self._uav_pos,
            num_tx_ant=self.config.num_bs_ant,
            num_rx_ant=1,
            num_subcarriers=128,
            subcarrier_spacing_hz=30e3,
            tx_name="bs_tx",
            rx_name="uav_probe",
            tx_look_at_rx=False,
            rx_look_at_tx=True,
        )

        h_ret = self.compute_leg_cfr(
            tx_position=self._uav_pos,
            rx_position=self._bs_pos,
            num_tx_ant=1,
            num_rx_ant=self.config.num_bs_ant,
            num_subcarriers=128,
            subcarrier_spacing_hz=30e3,
            tx_name="uav_echo",
            rx_name="bs_rx",
            tx_look_at_rx=True,
            rx_look_at_tx=False,
        )[:, 0, :]

        return {
            "model_type": self.config.model_type.value,
            "max_depth": self.config.max_depth,
            "specular_reflection": self.config.specular_reflection,
            "diffuse_scattering": self.config.diffuse_scattering,
            "bs_position": self._bs_pos.tolist(),
            "uav_position": self._uav_pos.tolist(),
            "target_range_m": self.config.get_target_range_m(),
            "target_doa_deg": self.config.get_target_doa_deg(),
            "h_out_power_db": float(
                10 * np.log10(np.mean(np.abs(h_out) ** 2) + 1e-12)
            ),
            "h_ret_power_db": float(
                10 * np.log10(np.mean(np.abs(h_ret) ** 2) + 1e-12)
            ),
        }

    def summary(self) -> str:
        uav_pos = self._uav_pos
        return (
            f"SionnaRTChannel(model={self.config.model_type.value}, "
            f"fc={self.config.carrier_frequency_hz / 1e9:.2f} GHz, "
            f"BS={self._bs_pos.tolist()}, "
            f"UAV=({uav_pos[0]:.1f}, {uav_pos[1]:.1f}, {uav_pos[2]:.1f}), "
            f"range={self.config.get_target_range_m():.1f}m, "
            f"DOA={self.config.get_target_doa_deg():.1f}deg, "
            f"max_depth={self.config.max_depth}, "
            f"specular={self.config.specular_reflection}, "
            f"diffuse={self.config.diffuse_scattering})"
        )