"""Configuration and result data structures for end-to-end ISAC simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from channel import ChannelModelType, SionnaRTConfig
from waveform import IsacWaveformConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "end2end"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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