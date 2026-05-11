"""Channel simulation module.

Provides:
- Sionna RT based ray-tracing channel models (LoS/NLoS/mixed)
- Multipath channel models (geometric/statistical)
- 3GPP channel models (UMi/UMa/RMa)
- Channel factory for unified access to different channel models
"""

from .sionna_rt_channel import (
    SionnaRTChannel,
    SionnaRTConfig,
    ChannelModelType,
    generate_cfr_dataset,
)
from .multipath_channel import (
    MultipathChannel,
    MultipathConfig,
    GeometricChannel,
    StatisticalChannel,
    FadingType,
    generate_multipath_cir,
)
from .ula_scene import UlaSceneConfig, simulate_ula_snapshots

__all__ = [
    # Sionna RT channel
    "SionnaRTChannel",
    "SionnaRTConfig",
    "ChannelModelType",
    "generate_cfr_dataset",
    # Multipath channel
    "MultipathChannel",
    "MultipathConfig",
    "GeometricChannel",
    "StatisticalChannel",
    "generate_multipath_cir",
    # Legacy ULA scene
    "UlaSceneConfig",
    "simulate_ula_snapshots",
]