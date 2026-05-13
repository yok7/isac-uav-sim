# channel/rt_config.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from sionna.rt import scene as rt_scene


C = 299_792_458.0


class ChannelModelType(Enum):
    """Channel model types for ISAC scenarios."""

    LOS = "los"
    NLOS = "nlos"
    MIXED = "mixed"
    MULTIPATH = "multipath"
    STREET_CANYON = "street_canyon"
    URBAN_MACRO = "urban_macro"


@dataclass
class SionnaRTConfig:
    """Configuration for Sionna RT ray-tracing channel."""

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

    def __post_init__(self) -> None:
        if self.scene_path is None:
            self.scene_path = self._get_default_scene()

    def _get_default_scene(self) -> str:
        if self.model_type == ChannelModelType.STREET_CANYON:
            return rt_scene.simple_street_canyon
        if self.model_type == ChannelModelType.URBAN_MACRO:
            return rt_scene.munich
        return rt_scene.simple_street_canyon

    def get_uav_position(self) -> np.ndarray:
        bearing_rad = np.deg2rad(self.uav_bearing_deg)
        x = self.bs_position[0] + self.uav_distance_m * np.cos(bearing_rad)
        y = self.bs_position[1] + self.uav_distance_m * np.sin(bearing_rad)
        z = self.uav_height_m
        return np.array([x, y, z], dtype=float)

    def get_target_doa_deg(self) -> float:
        return -self.uav_bearing_deg

    def get_target_range_m(self) -> float:
        rel = self.get_uav_position() - np.array(self.bs_position, dtype=float)
        return float(np.linalg.norm(rel))


def get_scene_path(config: SionnaRTConfig) -> str:
    if config.scene_path:
        return config.scene_path
    return rt_scene.simple_street_canyon