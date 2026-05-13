"""Channel simulation module.

Provides Sionna RT based ray-tracing channel models (LoS/NLoS/mixed)
for the ISAC end-to-end simulation framework.

Sub-modules:
  rt_config  — ChannelModelType / SionnaRTConfig
  rt_core    — RT scene, solver options, single-leg CFR
  rt_echo    — single-frame and multi-frame echo simulation
  rt_dataset — CFR dataset generation for training
  sionna_rt_channel — backwards-compatible facade
"""

from channel.sionna_rt_channel import (
    ChannelModelType,
    SionnaRTConfig,
    SionnaRTChannel,
    generate_cfr_dataset,
    generate_cir_dataset,
)

__all__ = [
    "ChannelModelType",
    "SionnaRTConfig",
    "SionnaRTChannel",
    "generate_cfr_dataset",
    "generate_cir_dataset",
]
