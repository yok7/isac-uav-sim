"""ISAC waveform generation using Sionna 5G NR PUSCH OFDM.

This module provides:
1. SionnaWaveformGenerator: Full 5G NR PUSCH OFDM waveform generation with channel estimation
2. IsacWaveformConfig: Configuration dataclass for ISAC waveform parameters
3. estimate_channel_ls: Least-squares channel estimation from DMRS
4. equalize_ofdm_symbols: MRC-based OFDM symbol equalization

Reference: Sionna 5G NR PUSCH tutorial + Link_Level_Simulations_with_RT tutorial
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver


@dataclass
class IsacWaveformConfig:
    """Configuration for ISAC OFDM waveform generation.

    Attributes:
        subcarrier_spacing_khz: Subcarrier spacing in kHz (15/30/60 kHz for 5G NR)
        num_rbs: Number of resource blocks (controls bandwidth)
        num_pusch_symbols: Number of PUSCH OFDM symbols per slot (10 or 14)
        num_tx_ant: Number of transmit antennas
        num_layers: Number of MIMO layers
        num_rx_ant: Number of receive antennas (for reference)
        dmrs_additional_position: DMRS additional position (0/1/2)
        dmrs_config_type: DMRS configuration type (1 or 2)
        mcs_index: MCS index for transport block (0-28, 14 = moderate)
        carrier_frequency_hz: Carrier frequency in Hz (default 3.5 GHz for sub-6G)
        comm_ratio: Communication resource fraction (0-1)
    """

    subcarrier_spacing_khz: float = 30.0
    num_rbs: int = 18
    num_pusch_symbols: int = 14
    num_tx_ant: int = 4
    num_layers: int = 2
    num_rx_ant: int = 8
    dmrs_additional_position: int = 0
    dmrs_config_type: int = 1
    mcs_index: int = 14
    carrier_frequency_hz: float = 3.5e9
    comm_ratio: float = 0.6

    def __post_init__(self):
        if self.subcarrier_spacing_khz not in [15, 30, 60]:
            raise ValueError(f"Invalid SCS {self.subcarrier_spacing_khz} kHz, must be 15/30/60")
        if self.num_pusch_symbols not in [10, 14]:
            raise ValueError(f"Invalid PUSCH symbols {self.num_pusch_symbols}, must be 10 or 14")


def to_numpy(x) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def estimate_channel_ls(
    y_pilot: np.ndarray,
    x_pilot: np.ndarray,
) -> np.ndarray:
    """Estimate OFDM channel using least-squares from DMRS pilots.

    Args:
        y_pilot: Received pilot symbols [num_rx_ant, num_pilot_re] complex
        x_pilot: Transmitted pilot symbols [num_pilot_re] complex (single stream)

    Returns:
        h_est: Estimated channel [num_rx_ant] complex (per-antenna scalar channel)
    """
    if x_pilot.size == 0:
        return np.zeros(y_pilot.shape[0], dtype=np.complex128)

    pilot_power = np.mean(np.abs(x_pilot) ** 2)
    if pilot_power < 1e-10:
        return np.zeros(y_pilot.shape[0], dtype=np.complex128)

    # LS estimate: h = y / x (element-wise)
    # y_pilot: [num_rx_ant, num_pilot_re]
    # x_pilot: [num_pilot_re]
    h_ls = np.mean(y_pilot / x_pilot[None, :], axis=1)
    return h_ls


def equalize_ofdm_symbols(
    y_symbols: np.ndarray,
    h_est: np.ndarray,
) -> np.ndarray:
    """Equalize OFDM symbols using MRC (Maximum Ratio Combining).

    Args:
        y_symbols: Received symbols [num_rx_ant, num_subcarriers] complex
        h_est: Estimated channel [num_rx_ant] complex

    Returns:
        y_eq: Equalized symbols [num_subcarriers] complex (beamformed scalar)
    """
    denom = np.sum(np.abs(h_est) ** 2)
    if denom < 1e-12:
        return np.zeros(y_symbols.shape[1], dtype=np.complex128)

    # MRC: y_eq = sum(conj(h_i) * y_i) / sum(|h_i|^2)
    return np.sum(np.conj(h_est)[:, None] * y_symbols, axis=0) / denom


class SionnaWaveformGenerator:
    """5G NR PUSCH OFDM waveform generator with Sionna.

    Provides full ISAC waveform generation with:
    - 5G NR compliant PUSCH OFDM modulation
    - Configurable DMRS pilot patterns
    - Built-in channel estimation from DMRS
    - Support for MIMO configurations

    Example:
        >>> config = IsacWaveformConfig(num_tx_ant=4, num_layers=2, num_rx_ant=8)
        >>> gen = SionnaWaveformGenerator(config)
        >>> x_freq, bits, rg = gen.generate_waveform(batch_size=32)
        >>> h_est = gen.estimate_channel_from_pilots(y_received, x_freq)
    """

    def __init__(
        self,
        config: IsacWaveformConfig,
        device: str | None = None,
        seed: int | None = 42,
    ):
        """Initialize the waveform generator.

        Args:
            config: ISAC waveform configuration
            device: Computation device ("cuda:0" or "cpu"), auto-detect if None
            seed: Random seed for reproducibility
        """
        self.config = config

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)
            self.seed = seed

        self._pusch_config = self._build_pusch_config()
        self._tx_freq = PUSCHTransmitter(
            self._pusch_config,
            output_domain="freq",
            return_bits=True,
            device=self.device,
        )
        self._tx_time = PUSCHTransmitter(
            self._pusch_config,
            output_domain="time",
            return_bits=False,
            device=self.device,
        )

        self._dmrs_mask: np.ndarray | None = None
        self._valid_mask: np.ndarray | None = None
        self._subcarrier_spacing_hz: float | None = None
        self._f_k: np.ndarray | None = None

    def _build_pusch_config(self) -> PUSCHConfig:
        """Build Sionna PUSCH configuration from IsacWaveformConfig."""
        cfg = PUSCHConfig()

        cfg.carrier.subcarrier_spacing = self.config.subcarrier_spacing_khz
        cfg.carrier.n_size_grid = self.config.num_rbs

        cfg.symbol_allocation = [0, self.config.num_pusch_symbols]

        cfg.num_antenna_ports = self.config.num_tx_ant
        cfg.num_layers = self.config.num_layers

        if self.config.num_layers == 1:
            cfg.precoding = "codebook"
            cfg.tpmi = 1
        else:
            cfg.precoding = "codebook"
            cfg.tpmi = 7

        cfg.dmrs.config_type = self.config.dmrs_config_type
        cfg.dmrs.length = 1
        cfg.dmrs.additional_position = self.config.dmrs_additional_position
        cfg.dmrs.num_cdm_groups_without_data = 2
        cfg.dmrs.dmrs_port_set = list(range(self.config.num_layers))

        cfg.tb.mcs_index = self.config.mcs_index
        cfg.tb.mcs_table = 1

        cfg.check_config()
        return cfg

    @property
    def pusch_config(self) -> PUSCHConfig:
        """Get the underlying Sionna PUSCHConfig."""
        return self._pusch_config

    @property
    def resource_grid(self):
        """Get the OFDM resource grid from the transmitter."""
        return self._tx_freq.resource_grid

    def generate_waveform(
        self,
        batch_size: int = 1,
        output_domain: Literal["freq", "time"] = "freq",
        return_bits: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Generate ISAC OFDM waveform.

        Args:
            batch_size: Number of waveforms to generate
            output_domain: "freq" for OFDM grid, "time" for time-domain signal
            return_bits: Whether to return transmitted bits

        Returns:
            x_freq: Transmitted signal [batch, num_tx, num_tx_ant, num_symbols, num_subcarriers]
            bits: Transmitted bits (if return_bits=True)
            valid_mask: Boolean mask indicating valid PUSCH RE [num_symbols, num_subcarriers]
        """
        if output_domain == "freq":
            # Sionna 2.0.1: PUSCHTransmitter takes inputs as positional arg;
            # return_bits is set at init, not at call time
            x_torch, bits = self._tx_freq(batch_size)
        else:
            x_torch = self._tx_time(None)
            bits = None

        # x shape: [batch, num_tx, num_tx_ant, num_symbols, num_subcarriers]
        x = to_numpy(x_torch)

        num_symbols = x.shape[3]
        num_subcarriers = x.shape[4]

        # Sionna DMRS mask shape is [subcarriers, symbols], transpose to [symbols, subcarriers]
        dmrs_mask_raw = to_numpy(self._pusch_config.dmrs_mask)
        dmrs_mask = dmrs_mask_raw.T.astype(bool)  # [symbols, subcarriers]

        # Clip to actual PUSCH allocation
        self._dmrs_mask = dmrs_mask[:num_symbols, :num_subcarriers]
        self._valid_mask = self._dmrs_mask.copy()

        # Store subcarrier frequencies for range-DOA estimation
        subcarrier_spacing_hz = self.config.subcarrier_spacing_khz * 1e3
        k_centered = np.arange(num_subcarriers) - (num_subcarriers - 1) / 2
        self._f_k = k_centered * subcarrier_spacing_hz
        self._subcarrier_spacing_hz = subcarrier_spacing_hz

        return x, bits, self._valid_mask

    def estimate_channel(
        self,
        y_received: np.ndarray,
        x_transmitted: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate channel from received DMRS pilots using LS.

        Args:
            y_received: Received signal [batch, num_rx_ant, num_symbols, num_subcarriers]
            x_transmitted: Transmitted signal [batch, num_tx, num_tx_ant, num_symbols, num_subcarriers]
            valid_mask: Valid RE mask (uses stored mask if None)

        Returns:
            h_est: Estimated channel [batch, num_rx_ant] complex
            pilot_snrs: Estimated SNR at pilots in dB
        """
        if valid_mask is None:
            valid_mask = self._valid_mask
        if valid_mask is None:
            raise ValueError("No valid mask available, call generate_waveform first")

        batch_size = y_received.shape[0]
        num_rx_ant = y_received.shape[1]

        h_est = np.zeros((batch_size, num_rx_ant), dtype=np.complex128)
        pilot_snrs = np.zeros(batch_size, dtype=float)

        for b in range(batch_size):
            y_batch = y_received[b]  # [num_rx_ant, num_symbols, num_subcarriers]
            x_batch = x_transmitted[b]  # [num_tx, num_tx_ant, num_symbols, num_subcarriers]

            # Extract pilot symbols: [num_rx_ant, num_pilot_re]
            pilot_y = y_batch[:, valid_mask]
            # Use first TX stream as reference for channel estimation
            x_pilot = x_batch[0, 0, valid_mask]  # [num_pilot_re]

            if x_pilot.size == 0:
                continue

            h_batch = estimate_channel_ls(pilot_y, x_pilot)
            h_est[b] = h_batch

            # Estimate pilot SNR
            sig_power = np.mean(np.abs(pilot_y) ** 2)
            noise_est = np.var(pilot_y - h_batch[:, None] * x_pilot[None, :])
            if noise_est > 0 and sig_power > noise_est:
                pilot_snrs[b] = 10 * np.log10(sig_power / noise_est)

        return h_est, pilot_snrs

    def compute_comm_rate(
        self,
        snr_db: float,
        pilot_fraction: float = 0.1,
    ) -> float:
        """Compute effective communication rate.

        Args:
            snr_db: SNR in dB
            pilot_fraction: Fraction of resources used for pilots

        Returns:
            Rate in bps/Hz
        """
        snr_lin = 10.0 ** (snr_db / 10.0)
        shannon = np.log2(1.0 + snr_lin)

        mcs_order = min(self.config.mcs_index + 1, 16)
        rate_ceiling = np.log2(max(mcs_order, 2))
        spectral_eff = min(shannon, rate_ceiling)

        payload_factor = max(0.0, 1.0 - 0.5 * pilot_fraction) * self.config.comm_ratio
        return float(payload_factor * spectral_eff)

    def get_dmrs_fraction(self) -> float:
        """Get the DMRS resource fraction in PUSCH allocation."""
        if self._valid_mask is None:
            return 0.0
        num_symbols = self._valid_mask.shape[0]
        num_subcarriers = self._valid_mask.shape[1]
        total_re = num_symbols * num_subcarriers
        return float(self._valid_mask.sum() / total_re) if total_re > 0 else 0.0

    def get_occupied_bandwidth_hz(self) -> float:
        """Get occupied bandwidth in Hz."""
        return 12 * self.config.num_rbs * self.config.subcarrier_spacing_khz * 1e3

    def beamform_receive(
        self,
        y: np.ndarray,
        angle_deg: float,
    ) -> np.ndarray:
        """Apply receive beamforming toward given angle.

        Args:
            y: Received signal [num_rx_ant, num_symbols, num_subcarriers]
            angle_deg: Beamforming direction in degrees

        Returns:
            y_bf: Beamformed signal [num_symbols, num_subcarriers]
        """
        num_rx_ant = y.shape[0]
        theta = np.deg2rad(angle_deg)
        n = np.arange(num_rx_ant)
        a = np.exp(-1j * 2 * np.pi * 0.5 * n * np.sin(theta))

        a_norm = a / np.sqrt(num_rx_ant)
        y_bf = np.einsum("r,rsk->sk", np.conj(a_norm), y)
        return y_bf

    def summary(self) -> str:
        """Return a summary string of the generator configuration."""
        return (
            f"SionnaWaveformGenerator(config=IsacWaveformConfig(\n"
            f"  subcarrier_spacing_khz={self.config.subcarrier_spacing_khz},\n"
            f"  num_rbs={self.config.num_rbs},\n"
            f"  num_pusch_symbols={self.config.num_pusch_symbols},\n"
            f"  num_tx_ant={self.config.num_tx_ant},\n"
            f"  num_layers={self.config.num_layers},\n"
            f"  dmrs_additional_position={self.config.dmrs_additional_position},\n"
            f"  mcs_index={self.config.mcs_index},\n"
            f"  carrier_frequency_hz={self.config.carrier_frequency_hz/1e9:.2f} GHz),\n"
            f"  device={self.device})"
        )