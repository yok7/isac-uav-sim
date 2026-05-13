"""Public facade for the split Sionna RT channel implementation.

This file keeps the old public API stable:

    from channel import SionnaRTChannel, SionnaRTConfig, ChannelModelType

Internally, the implementation is split into:

    rt_config.py   -> ChannelModelType / SionnaRTConfig
    rt_core.py     -> Sionna RT scene setup and one-leg CFR computation
    rt_echo.py     -> single-frame and multi-frame ISAC echo synthesis
    rt_dataset.py  -> CFR dataset generation

External modules should normally import from channel, not from the internal
split files directly.
"""

from __future__ import annotations

from channel.rt_config import ChannelModelType, SionnaRTConfig
from channel.rt_core import SionnaRTChannelCore
from channel.rt_echo import RTEchoMixin
from channel.rt_dataset import generate_cfr_dataset, generate_cir_dataset


class SionnaRTChannel(RTEchoMixin, SionnaRTChannelCore):
    """Sionna RT channel with ISAC echo synthesis.

    This class is intentionally thin. It combines:

        SionnaRTChannelCore:
            Sionna RT scene construction and one-leg CFR computation.

        RTEchoMixin:
            ISAC echo construction, including:
                BS -> UAV propagation
                UAV point-target reflection
                UAV -> BS return propagation
                AWGN
                optional Doppler phase
                multi-frame slow-time echo generation

    The public behavior remains compatible with the previous monolithic
    sionna_rt_channel.py implementation.
    """

    pass


__all__ = [
    "ChannelModelType",
    "SionnaRTConfig",
    "SionnaRTChannel",
    "generate_cfr_dataset",
    "generate_cir_dataset",
]