"""ISAC waveform generation module.

Provides:
- Sionna-based 5G NR PUSCH OFDM waveform generation with channel estimation
- Simplified OFDM waveforms for fast prototyping
- Resource allocation utilities for ISAC tradeoffs
"""

from .isac_waveform import (
    SionnaWaveformGenerator,
    IsacWaveformConfig,
    estimate_channel_ls,
    equalize_ofdm_symbols,
)
from .isac_ofdm import (
    IsacOfdmConfig,
    ber_proxy_qpsk,
    effective_comm_rate_bpshz,
    generate_isac_ofdm_frame,
)

__all__ = [
    # Sionna-based waveforms
    "SionnaWaveformGenerator",
    "IsacWaveformConfig",
    "estimate_channel_ls",
    "equalize_ofdm_symbols",
    # Simplified waveforms
    "IsacOfdmConfig",
    "generate_isac_ofdm_frame",
    "effective_comm_rate_bpshz",
    "ber_proxy_qpsk",
]