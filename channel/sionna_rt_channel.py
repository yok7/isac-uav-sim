"""Sionna RT based ray-tracing channel simulation.

This module provides:
1. SionnaRTConfig: Configuration for different channel scenarios (LoS/NLoS/Mixed/Multipath)
2. SionnaRTChannel: Ray-tracing channel using Sionna RT with two-leg propagation

Supported scenarios:
- LoS: Line-of-Sight only (single path)
- NLoS: Non-Line-of-Sight with specular/diffuse reflections
- Mixed: Both LoS and NLoS paths
- Multipath: Multiple reflection paths (urban canyon)
- StreetCanyon: Urban street canyon (Sionna RT simple_street_canyon)

Physical model:
- BS (4 Tx) → UAV point target (1 Rx)
- UAV point target (1 Tx) → BS (8 Rx)
- Monostatic ISAC echo: combines two propagation legs with point-target reflection

Reference:
- Sionna RT tutorials: Link_Level_Simulations_with_RT.ipynb
- 3GPP TR 38.901 channel model
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import torch

from sionna.rt import (
    load_scene,
    PathSolver,
    PlanarArray,
    Transmitter,
    Receiver,
    scene as rt_scene,
    subcarrier_frequencies,
)


C = 299_792_458.0


class ChannelModelType(Enum):
    """Channel model types for ISAC scenarios."""
    LOS = "los"                    # Line-of-Sight only
    NLOS = "nlos"                  # Non-Line-of-Sight with reflections
    MIXED = "mixed"                # Both LoS and NLoS paths
    MULTIPATH = "multipath"         # Multiple reflection paths (urban canyon)
    STREET_CANYON = "street_canyon" # Urban street canyon (Sionna RT simple_street_canyon)
    URBAN_MACRO = "urban_macro"    # Munich/Urban macro (Sionna RT city scenes)


@dataclass
class SionnaRTConfig:
    """Configuration for Sionna RT ray-tracing channel.

    Attributes:
        model_type: Type of channel model (LoS/NLoS/Mixed/Multipath/StreetCanyon)
        carrier_frequency_hz: Carrier frequency in Hz
        max_depth: Maximum reflection depth for ray tracing (>0 enables multipath)
        samples_per_src: Number of samples per source for ray tracing
        min_gain_db: Minimum path gain to consider (dB)
        bs_position: Base station position [x, y, z] in meters
        uav_height_m: UAV target height in meters
        uav_distance_m: Horizontal distance from BS to UAV in meters
        uav_bearing_deg: UAV bearing angle from BS in degrees
        num_bs_ant: Number of BS antennas (ULA: num_cols, UPA: num_rows*num_cols)
        specular_reflection: Enable specular reflection (creates multipath)
        diffuse_scattering: Enable diffuse scattering (creates more multipath)
        scene_path: Custom scene path, overrides model_type
    """

    model_type: ChannelModelType = ChannelModelType.STREET_CANYON
    carrier_frequency_hz: float = 3.5e9
    max_depth: int = 3
    samples_per_src: int = 5000
    min_gain_db: float = -130.0

    bs_position: tuple[float, float, float] = (0.0, 0.0, 25.0)
    uav_height_m: float = 35.0
    uav_distance_m: float = 130.0
    uav_bearing_deg: float = 20.0

    num_bs_ant: int = 8

    specular_reflection: bool = True
    diffuse_scattering: bool = True

    scene_path: str | None = None

    def __post_init__(self):
        if self.scene_path is None:
            self.scene_path = self._get_default_scene()

    def _get_default_scene(self) -> str:
        """Get default scene path based on model type."""
        if self.model_type == ChannelModelType.STREET_CANYON:
            return rt_scene.simple_street_canyon
        elif self.model_type == ChannelModelType.URBAN_MACRO:
            return rt_scene.munich
        else:
            return rt_scene.simple_street_canyon

    def get_uav_position(self) -> np.ndarray:
        """Compute UAV position from bearing and distance.

        Uses the PPT-specified target position [120, 45, 35] m as reference.
        """
        bearing_rad = np.deg2rad(self.uav_bearing_deg)
        x = self.bs_position[0] + self.uav_distance_m * np.cos(bearing_rad)
        y = self.bs_position[1] + self.uav_distance_m * np.sin(bearing_rad)
        z = self.uav_height_m
        return np.array([x, y, z])

    def get_target_doa_deg(self) -> float:
        """Compute target DOA angle from BS perspective."""
        return -self.uav_bearing_deg

    def get_target_range_m(self) -> float:
        """Compute target range from BS."""
        rel = self.get_uav_position() - np.array(self.bs_position)
        return float(np.linalg.norm(rel))


def _get_scene_path(config: SionnaRTConfig) -> str:
    """Resolve scene path from config."""
    if config.scene_path:
        return config.scene_path
    return rt_scene.simple_street_canyon


class SionnaRTChannel:
    """Ray-tracing channel simulator using Sionna RT.

    Provides realistic radio propagation simulation with two-leg propagation:
    - Leg 1: BS (num_tx_ant) → UAV target
    - Leg 2: UAV target → BS (num_rx_ant)

    Monostatic ISAC echo combines both legs with point-target reflection.

    Example:
        >>> config = SionnaRTConfig(model_type=ChannelModelType.STREET_CANYON)
        >>> channel = SionnaRTChannel(config)
        >>> rng = np.random.default_rng(42)
        >>> x_tx = np.random.randn(1, 4, 14, 128) + 1j*np.random.randn(1, 4, 14, 128)
        >>> valid_mask = np.ones((14, 128), dtype=bool)
        >>> y_echo, h_out, h_ret = channel.simulate_echo(
        ...     x_tx=x_tx,
        ...     valid_mask=valid_mask,
        ...     subcarrier_spacing_hz=30e3,
        ...     snr_db=10.0,
        ...     rng=rng,
        ... )
        >>> # y_echo shape: [batch=1, num_rx_ant, num_symbols, num_subcarriers]
    """

    def __init__(
        self,
        config: SionnaRTConfig,
        device: str | None = None,
        seed: int | None = 42,
    ):
        """Initialize Sionna RT channel.

        Args:
            config: Channel configuration
            device: Computation device ("cuda:0" or "cpu")
            seed: Random seed for reproducibility
        """
        self.config = config

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.seed = seed
        self._path_solver = PathSolver()

        self._bs_pos = np.array(config.bs_position)
        self._uav_pos = config.get_uav_position()

    def _get_solver_options(self) -> dict:
        """Get PathSolver options based on channel model type."""
        model_type = self.config.model_type

        if model_type == ChannelModelType.LOS:
            return {
                "los": True,
                "specular_reflection": False,
                "diffuse_reflection": False,
                "refraction": False,
                "diffraction": False,
            }
        elif model_type in [ChannelModelType.NLOS, ChannelModelType.MIXED]:
            return {
                "los": model_type == ChannelModelType.MIXED,
                "specular_reflection": self.config.specular_reflection,
                "diffuse_reflection": self.config.diffuse_scattering,
                "refraction": False,
                "diffraction": False,
            }
        elif model_type in [ChannelModelType.MULTIPATH, ChannelModelType.STREET_CANYON]:
            return {
                "los": True,
                "specular_reflection": self.config.specular_reflection,
                "diffuse_reflection": self.config.diffuse_scattering,
                "refraction": False,
                "diffraction": False,
            }
        else:
            return {
                "los": True,
                "specular_reflection": True,
                "diffuse_reflection": self.config.diffuse_scattering,
                "refraction": False,
                "diffraction": False,
            }

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
        """
        Compute one RT propagation leg (TX → RX).

        Args:
            tx_position: Transmitter position [x, y, z] in meters
            rx_position: Receiver position [x, y, z] in meters
            num_tx_ant: Number of transmit antenna elements (ULA: num_cols)
            num_rx_ant: Number of receive antenna elements
            num_subcarriers: Number of OFDM subcarriers
            subcarrier_spacing_hz: Subcarrier spacing in Hz
            tx_name: Name for transmitter in scene
            rx_name: Name for receiver in scene
            tx_look_at_rx: Whether to point TX toward RX
            rx_look_at_tx: Whether to point RX toward TX
            seed: Random seed override

        Returns:
            h: [num_rx_ant, num_tx_ant, num_subcarriers] complex CFR tensor
        """
        scene = load_scene(_get_scene_path(self.config))
        scene.frequency = self.config.carrier_frequency_hz

        scene.tx_array = PlanarArray(
            num_rows=1,
            num_cols=num_tx_ant,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )
        scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=num_rx_ant,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )

        tx_kwargs = {"look_at": rx_position.tolist()} if tx_look_at_rx else {"orientation": [0.0, 0.0, 0.0]}
        rx_kwargs = {"look_at": tx_position.tolist()} if rx_look_at_tx else {"orientation": [0.0, 0.0, 0.0]}

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

        solver_opts = self._get_solver_options()

        paths = self._path_solver(
            scene,
            max_depth=self.config.max_depth,
            samples_per_src=self.config.samples_per_src,
            synthetic_array=False,
            seed=seed if seed is not None else self.seed,
            **solver_opts,
        )

        freqs = subcarrier_frequencies(num_subcarriers, subcarrier_spacing_hz)
        h = paths.cfr(
            freqs,
            normalize=False,
            normalize_delays=False,
            out_type="numpy",
        )

        # Sionna RT CFR shape: [num_rx, rx_ant, num_tx, tx_ant, num_time_steps, num_subcarriers]
        # Extract: [num_rx_ant, num_tx_ant, num_subcarriers]
        return np.asarray(h[0, :, 0, :, 0, :], dtype=np.complex128)

    def simulate_echo(
        self,
        x_tx: np.ndarray,
        valid_mask: np.ndarray,
        subcarrier_spacing_hz: float,
        snr_db: float,
        rng: np.random.Generator,
        num_rx_ant: int = 8,
        target_velocity_mps: np.ndarray | None = None,
        ofdm_symbol_duration_s: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate monostatic ISAC echo with two-leg RT propagation.

        Physical model:
        - BS (num_tx_ant) transmits OFDM → UAV target
        - UAV point target reflects with complex coefficient → BS (num_rx_ant)

        Args:
            x_tx: Transmitted OFDM signal [batch, num_tx_ant, num_symbols, num_subcarriers]
            valid_mask: Valid RE mask [num_symbols, num_subcarriers]
            subcarrier_spacing_hz: Subcarrier spacing in Hz
            snr_db: Sensing SNR in dB
            rng: Random number generator
            num_rx_ant: Number of BS receive antennas (default 8)
            target_velocity_mps: UAV velocity vector [vx, vy, vz] in m/s for Doppler
            ofdm_symbol_duration_s: OFDM symbol duration (including CP)
                If None, uses 1 / subcarrier_spacing_hz as approximation

        Returns:
            y_echo: Received echo [batch, num_rx_ant, num_symbols, num_subcarriers]
            h_out: BS→UAV CFR [num_tx_ant, num_subcarriers]
            h_ret: UAV→BS CFR [num_rx_ant, num_subcarriers]
        """
        if x_tx.ndim != 4:
            raise ValueError(
                f"x_tx must have shape [batch, num_tx_ant, num_symbols, num_subcarriers], "
                f"got shape {x_tx.shape}"
            )

        batch_size, num_tx_ant, num_symbols, num_subcarriers = x_tx.shape

        bs_pos = self._bs_pos
        uav_pos = self._uav_pos

        # Leg 1: BS → UAV
        h_out = self.compute_leg_cfr(
            tx_position=bs_pos,
            rx_position=uav_pos,
            num_tx_ant=num_tx_ant,
            num_rx_ant=1,
            num_subcarriers=num_subcarriers,
            subcarrier_spacing_hz=subcarrier_spacing_hz,
            tx_name="bs_tx",
            rx_name="uav_probe",
            tx_look_at_rx=False,
            rx_look_at_tx=True,
            seed=self.seed,
        )[0, :, :]  # [num_tx_ant, num_subcarriers]

        # Leg 2: UAV → BS
        h_ret = self.compute_leg_cfr(
            tx_position=uav_pos,
            rx_position=bs_pos,
            num_tx_ant=1,
            num_rx_ant=num_rx_ant,
            num_subcarriers=num_subcarriers,
            subcarrier_spacing_hz=subcarrier_spacing_hz,
            tx_name="uav_echo",
            rx_name="bs_rx",
            tx_look_at_rx=True,
            rx_look_at_tx=False,
            seed=self.seed,
        )[:, 0, :]  # [num_rx_ant, num_subcarriers]

        # BS → UAV: incident field on target
        # h_out: [num_tx_ant, num_subcarriers]
        # x_tx:  [batch, num_tx_ant, num_symbols, num_subcarriers]
        # target_field: [batch, num_symbols, num_subcarriers]
        target_field = np.einsum("tk,btnk->bnk", h_out, x_tx)

        # Point-target reflection coefficient (random phase for non-coherent target)
        reflection_coeff = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))

        # UAV → BS: received echo at BS array
        # h_ret: [num_rx_ant, num_subcarriers]
        # echo_signal: [batch, num_rx_ant, num_symbols, num_subcarriers]
        echo_signal = (
            reflection_coeff
            * h_ret[None, :, None, :]
            * target_field[:, None, :, :]
        )

        # Doppler for moving target
        if target_velocity_mps is not None:
            rel = uav_pos - bs_pos
            distance = np.linalg.norm(rel) + 1e-12
            unit_los = rel / distance
            radial_velocity = float(np.dot(target_velocity_mps, unit_los))

            wavelength = C / self.config.carrier_frequency_hz
            doppler_hz = 2.0 * radial_velocity / wavelength

            if ofdm_symbol_duration_s is None:
                ofdm_symbol_duration_s = 1.0 / subcarrier_spacing_hz

            t_sym = np.arange(num_symbols) * ofdm_symbol_duration_s
            doppler_phase = np.exp(1j * 2.0 * np.pi * doppler_hz * t_sym)
            echo_signal = echo_signal * doppler_phase[None, None, :, None]

        # Add AWGN noise
        selected_power = np.mean(np.abs(echo_signal[:, :, valid_mask]) ** 2)
        noise_power = selected_power / (10 ** (snr_db / 10.0))
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal(echo_signal.shape)
            + 1j * rng.standard_normal(echo_signal.shape)
        )

        return echo_signal + noise, h_out, h_ret

    def simulate_echo_multi_frame(
        self,
        x_tx: np.ndarray,
        valid_mask: np.ndarray,
        subcarrier_spacing_hz: float,
        snr_db: float,
        num_frames: int,
        frame_interval_s: float,
        initial_uav_pos: np.ndarray,
        target_velocity_mps: np.ndarray,
        bs_pos: np.ndarray,
        rng: np.random.Generator,
        num_rx_ant: int = 8,
        update_channel_every: int | None = None,
    ) -> np.ndarray:
        """Simulate coherent multi-frame ISAC echo for radial velocity estimation.

        For the first velocity-estimation version, the RT geometry is kept fixed
        within one coherent processing interval. Motion is represented by a
        deterministic slow-time Doppler phase.

        Args:
            x_tx: Transmitted OFDM signal [num_tx_ant, num_symbols, num_subcarriers]
            valid_mask: Valid RE mask [num_symbols, num_subcarriers]
            subcarrier_spacing_hz: Subcarrier spacing in Hz
            snr_db: Sensing SNR in dB
            num_frames: Number of slow-time frames
            frame_interval_s: Time interval between consecutive frames (seconds)
            initial_uav_pos: Initial UAV position [x, y, z] in meters
            target_velocity_mps: UAV velocity vector [vx, vy, vz] in m/s
            bs_pos: Base station position [x, y, z] in meters
            rng: Random number generator
            num_rx_ant: Number of BS receive antennas
            update_channel_every: Re-run RT channel every N frames.
                Default: None (= num_frames, i.e., fixed geometry for full CPI)

        Returns:
            y_echo_frames: Received echo [num_frames, num_rx_ant, num_symbols, num_subcarriers]
        """
        if x_tx.ndim != 3:
            raise ValueError(
                f"x_tx must be [num_tx_ant, num_symbols, num_subcarriers], got {x_tx.shape}"
            )

        num_tx_ant, num_symbols, num_subcarriers = x_tx.shape

        if update_channel_every is None:
            update_channel_every = num_frames

        bs_pos = np.asarray(bs_pos, dtype=float)
        uav_pos0 = np.asarray(initial_uav_pos, dtype=float)
        vel = np.asarray(target_velocity_mps, dtype=float)

        y_echo_frames = np.zeros(
            (num_frames, num_rx_ant, num_symbols, num_subcarriers),
            dtype=np.complex128,
        )

        # Keep one coherent target reflection coefficient within one CPI.
        reflection_coeff = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))

        # Compute initial RT channel once.
        current_uav_pos = uav_pos0.copy()
        h_out = self.compute_leg_cfr(
            tx_position=bs_pos,
            rx_position=current_uav_pos,
            num_tx_ant=num_tx_ant,
            num_rx_ant=1,
            num_subcarriers=num_subcarriers,
            subcarrier_spacing_hz=subcarrier_spacing_hz,
            tx_name="bs_tx",
            rx_name="uav_probe",
            tx_look_at_rx=False,
            rx_look_at_tx=True,
            seed=self.seed,
        )[0, :, :]  # [num_tx_ant, num_subcarriers]

        h_ret = self.compute_leg_cfr(
            tx_position=current_uav_pos,
            rx_position=bs_pos,
            num_tx_ant=1,
            num_rx_ant=num_rx_ant,
            num_subcarriers=num_subcarriers,
            subcarrier_spacing_hz=subcarrier_spacing_hz,
            tx_name="uav_echo",
            rx_name="bs_rx",
            tx_look_at_rx=True,
            rx_look_at_tx=False,
            seed=self.seed,
        )[:, 0, :]  # [num_rx_ant, num_subcarriers]

        # Base echo for one frame: [num_rx_ant, num_symbols, num_subcarriers]
        target_field = np.einsum("tk,tnk->nk", h_out, x_tx)
        base_echo = reflection_coeff * h_ret[:, None, :] * target_field[None, :, :]

        wavelength = C / self.config.carrier_frequency_hz

        # Use initial LOS direction for radial velocity in one CPI.
        rel0 = uav_pos0 - bs_pos
        unit_los0 = rel0 / (np.linalg.norm(rel0) + 1e-12)
        radial_velocity = float(np.dot(vel, unit_los0))
        doppler_hz = 2.0 * radial_velocity / wavelength

        mask_3d = valid_mask[np.newaxis, :, :]

        for frame_idx in range(num_frames):
            # Optional slow geometry update.
            if frame_idx > 0 and update_channel_every < num_frames and frame_idx % update_channel_every == 0:
                current_uav_pos = uav_pos0 + vel * (frame_idx * frame_interval_s)

                h_out = self.compute_leg_cfr(
                    tx_position=bs_pos,
                    rx_position=current_uav_pos,
                    num_tx_ant=num_tx_ant,
                    num_rx_ant=1,
                    num_subcarriers=num_subcarriers,
                    subcarrier_spacing_hz=subcarrier_spacing_hz,
                    tx_name="bs_tx",
                    rx_name="uav_probe",
                    tx_look_at_rx=False,
                    rx_look_at_tx=True,
                    seed=self.seed + frame_idx,
                )[0, :, :]

                h_ret = self.compute_leg_cfr(
                    tx_position=current_uav_pos,
                    rx_position=bs_pos,
                    num_tx_ant=1,
                    num_rx_ant=num_rx_ant,
                    num_subcarriers=num_subcarriers,
                    subcarrier_spacing_hz=subcarrier_spacing_hz,
                    tx_name="uav_echo",
                    rx_name="bs_rx",
                    tx_look_at_rx=True,
                    rx_look_at_tx=False,
                    seed=self.seed + frame_idx,
                )[:, 0, :]

                target_field = np.einsum("tk,tnk->nk", h_out, x_tx)
                base_echo = reflection_coeff * h_ret[:, None, :] * target_field[None, :, :]

            # Deterministic Doppler phase accumulation
            t = frame_idx * frame_interval_s
            doppler_phase = np.exp(1j * 2.0 * np.pi * doppler_hz * t)

            signal = base_echo * doppler_phase

            selected_power = np.sum(np.abs(signal) ** 2 * mask_3d) / np.count_nonzero(mask_3d)
            noise_power = selected_power / (10 ** (snr_db / 10.0))
            noise = np.sqrt(noise_power / 2.0) * (
                rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
            )

            y_echo_frames[frame_idx] = signal + noise

        return y_echo_frames

    def get_channel_stats(self) -> dict:
        """Get channel statistics (path gains, delays, angles) from both legs."""
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
            "h_out_power_db": float(10 * np.log10(np.mean(np.abs(h_out) ** 2) + 1e-12)),
            "h_ret_power_db": float(10 * np.log10(np.mean(np.abs(h_ret) ** 2) + 1e-12)),
        }

    def summary(self) -> str:
        """Return summary of channel configuration."""
        uav_pos = self._uav_pos
        target_range = self.config.get_target_range_m()
        target_doa = self.config.get_target_doa_deg()

        return (
            f"SionnaRTChannel(model={self.config.model_type.value}, "
            f"fc={self.config.carrier_frequency_hz/1e9:.2f} GHz, "
            f"BS={self._bs_pos.tolist()}, "
            f"UAV=({uav_pos[0]:.1f}, {uav_pos[1]:.1f}, {uav_pos[2]:.1f}), "
            f"range={target_range:.1f}m, DOA={target_doa:.1f}deg, "
            f"max_depth={self.config.max_depth}, "
            f"specular={self.config.specular_reflection}, "
            f"diffuse={self.config.diffuse_scattering})"
        )


def generate_cfr_dataset(
    config: SionnaRTConfig,
    num_cirs: int,
    batch_size: int = 100,
    device: str | None = None,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Generate a dataset of Channel Frequency Responses.

    Args:
        config: Sionna RT configuration
        num_cirs: Total number of CFRs to generate
        batch_size: CFRs per batch
        device: Computation device
        seed: Random seed base

    Returns:
        List of CFR arrays [num_rx_ant, num_tx_ant, num_subcarriers]

    Reference: Link_Level_Simulations_with_RT.ipynb
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    channel = SionnaRTChannel(config, device=device, seed=seed)

    cfr_list = []

    num_batches = int(np.ceil(num_cirs / batch_size))

    for batch_idx in range(num_batches):
        print(f"Generating CFR batch {batch_idx + 1}/{num_batches}...", end="\r")

        batch_cfr = channel.compute_leg_cfr(
            tx_position=channel._bs_pos,
            rx_position=channel._uav_pos + np.array([
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(-2, 2),
            ]),
            num_tx_ant=config.num_bs_ant,
            num_rx_ant=1,
            num_subcarriers=128,
            subcarrier_spacing_hz=30e3,
            tx_name="bs_tx",
            rx_name="uav_probe",
            seed=seed + batch_idx if seed else None,
        )

        cfr_list.append(batch_cfr)

    print(f"\nGenerated {num_cirs} CFRs")
    return cfr_list