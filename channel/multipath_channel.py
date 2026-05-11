"""Statistical multipath channel models for ablation baseline.

IMPORTANT: This module is for algorithm sanity check and ablation studies ONLY.
For the main experiments, use SionnaRTChannel which provides physically accurate
ray-tracing based multipath from real scene geometry.

This module generates simplified multipath channels with predefined delays, gains,
and DOAs. The paths are NOT derived from scene geometry - they are artificially
specified. Do NOT use this for "realistic multipath scenarios" in papers/reports.

Use cases:
- Rapid algorithm prototyping (fast, no GPU needed)
- Ablation studies (compare with/without multipath)
- Theoretical analysis with controlled parameters

For main experiments, use channel.SionnaRTChannel with:
  - model_type=ChannelModelType.STREET_CANYON
  - max_depth > 0
  - specular_reflection=True
  - diffuse_scattering=True

Reference: 3GPP TR 38.901 channel model (for parameter ranges)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FadingType(Enum):
    """Fading distribution types."""
    RAYLEIGH = "rayleigh"   # Pure multipath, no LoS
    RICIAN = "rician"        # LoS + multipath
    NAKAGAMI = "nakagami"    # Nakagami-m fading


@dataclass
class MultipathConfig:
    """Configuration for multipath channel.

    Attributes:
        num_elements: Number of antenna elements
        d_over_lambda: Element spacing relative to wavelength
        num_snapshots: Number of time snapshots
        num_paths: Number of multipath components
        snr_db: Signal-to-noise ratio in dB
        fading_type: Type of fading distribution
        rician_k_db: Rician K-factor in dB (for Rician fading)
        delay_spread_ns: RMS delay spread in nanoseconds
        angle_spread_deg: RMS angle spread in degrees
        doppler_hz: Maximum Doppler frequency in Hz
        los_doa_deg: LoS path DOA in degrees
        los_range_m: LoS path range in meters
        reflect_gains_db: Reflection gains for each path in dB
        reflect_doas_deg: DOA for each reflection path in degrees
        reflect_delays_ns: Delay for each reflection path in ns
    """

    num_elements: int = 8
    d_over_lambda: float = 0.5
    num_snapshots: int = 128
    num_paths: int = 4
    snr_db: float = 10.0

    fading_type: FadingType = FadingType.RICIAN
    rician_k_db: float = 6.0

    delay_spread_ns: float = 100.0
    angle_spread_deg: float = 10.0
    doppler_hz: float = 0.0

    los_doa_deg: float = 0.0
    los_range_m: float = 100.0

    reflect_gains_db: tuple[float, ...] = (-6.0, -8.0, -10.0, -12.0)
    reflect_doas_deg: tuple[float, ...] = (-20.0, 10.0, -5.0, 15.0)
    reflect_delays_ns: tuple[float, ...] = (50.0, 100.0, 150.0, 200.0)


def steering_vector_ula(
    num_elements: int,
    angle_deg: float,
    d_over_lambda: float = 0.5,
) -> np.ndarray:
    """Compute ULA steering vector."""
    theta = np.deg2rad(angle_deg)
    n = np.arange(num_elements)
    return np.exp(-1j * 2 * np.pi * d_over_lambda * n * np.sin(theta))


def generate_multipath_cir(
    config: MultipathConfig,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate multipath CIR from configuration.

    Args:
        config: Multipath channel configuration
        rng: Random number generator

    Returns:
        Dict with keys:
            - delays: Path delays in seconds
            - gains_linear: Path gains in linear scale
            - doas_deg: Path DOAs in degrees
            - a_mat: Array response matrix [num_elements, num_paths]
    """
    if rng is None:
        rng = np.random.default_rng()

    num_elements = config.num_elements
    num_paths = config.num_paths

    delays = np.array(config.reflect_delays_ns[:num_paths]) * 1e-9
    gains_db = np.array(config.reflect_gains_db[:num_paths])
    gains_linear = 10.0 ** (gains_db / 20.0)
    doas_deg = np.array(config.reflect_doas_deg[:num_paths])

    a_mat = np.zeros((num_elements, num_paths), dtype=np.complex128)
    for p in range(num_paths):
        a_mat[:, p] = steering_vector_ula(
            num_elements, doas_deg[p], config.d_over_lambda
        )

    rician_k = 10.0 ** (config.rician_k_db / 10.0)

    if config.fading_type == FadingType.RICIAN and rician_k > 0.01:
        los_gain = np.sqrt(rician_k / (1 + rician_k))
        multipath_gain = 1.0 / np.sqrt(1 + rician_k)
        doas_deg = np.concatenate([[config.los_doa_deg], doas_deg])
        delays = np.concatenate([[0.0], delays])
        gains_linear = np.concatenate([[los_gain], gains_linear * multipath_gain])
        a_mat = np.zeros((num_elements, num_paths + 1), dtype=np.complex128)
        a_mat[:, 0] = steering_vector_ula(num_elements, config.los_doa_deg, config.d_over_lambda)
        for p in range(num_paths):
            a_mat[:, p + 1] = steering_vector_ula(
                num_elements, config.reflect_doas_deg[p], config.d_over_lambda
            )

    return {
        "delays": delays,
        "gains_linear": gains_linear,
        "doas_deg": doas_deg,
        "a_mat": a_mat,
        "num_paths": len(delays),
    }


class MultipathChannel:
    """Statistical multipath channel simulator.

    Provides realistic multipath channel simulation with:
    - Configurable number of paths
    - Rician/Rayleigh fading
    - Delay and angle spread
    - Configurable Doppler for moving targets
    - Snapshot-based simulation

    Example:
        >>> config = MultipathConfig(num_paths=4, rician_k_db=6)
        >>> channel = MultipathChannel(config)
        >>> x, cov, true_doas = channel.simulate()
    """

    def __init__(
        self,
        config: MultipathConfig,
        seed: int | None = None,
    ):
        """Initialize multipath channel.

        Args:
            config: Multipath channel configuration
            seed: Random seed
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
        self._cir = generate_multipath_cir(config, self.rng)

    def simulate(
        self,
        num_snapshots: int | None = None,
        snr_db: float | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate array snapshots and covariance matrix.

        Args:
            num_snapshots: Number of snapshots (uses config if None)
            snr_db: SNR in dB (uses config if None)
            seed: Random seed override

        Returns:
            x: Array snapshots [num_snapshots, num_elements] complex
            cov: Sample covariance matrix [num_elements, num_elements]
            true_doas: True path DOAs in degrees
        """
        if num_snapshots is None:
            num_snapshots = self.config.num_snapshots
        if snr_db is None:
            snr_db = self.config.snr_db
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        m = self.config.num_elements
        n = num_snapshots

        x_signal = np.zeros((n, m), dtype=np.complex128)

        delays = self._cir["delays"]
        gains = self._cir["gains_linear"]
        doas_deg = self._cir["doas_deg"]
        a_mat = self._cir["a_mat"]
        num_paths = self._cir["num_paths"]

        for p in range(num_paths):
            a = a_mat[:, p]

            doppler_phase = 0.0
            if self.config.doppler_hz > 0:
                time_vec = np.arange(n) / n
                doppler = 2 * np.pi * self.config.doppler_hz * time_vec
                doppler_phase = rng.uniform(0, 2 * np.pi) + doppler

            s = gains[p] * (
                rng.standard_normal(n) + 1j * rng.standard_normal(n)
            ) / np.sqrt(2)
            s *= np.exp(1j * doppler_phase)

            x_signal += np.outer(s, np.conj(a))

        sig_power = float(np.mean(np.abs(x_signal) ** 2))
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_power = sig_power / max(snr_lin, 1e-12)
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
        )

        x = x_signal + noise
        cov = (x.conj().T @ x) / n

        unique_doas = []
        seen = set()
        for d in doas_deg:
            rounded = round(d, 1)
            if rounded not in seen:
                seen.add(rounded)
                unique_doas.append(d)
        true_doas = np.array(sorted(unique_doas), dtype=float)

        return x, cov, true_doas

    def get_channel_parameters(self) -> dict:
        """Get channel parameters (delays, gains, DOAs)."""
        return {
            "delays_ns": self._cir["delays"] * 1e9,
            "gains_db": 20 * np.log10(self._cir["gains_linear"] + 1e-12),
            "doas_deg": self._cir["doas_deg"],
            "num_paths": self._cir["num_paths"],
            "rms_delay_spread_ns": self._compute_delay_spread(),
        }

    def _compute_delay_spread(self) -> float:
        """Compute RMS delay spread."""
        delays = self._cir["delays"]
        gains = self._cir["gains_linear"] ** 2
        gains = gains / (np.sum(gains) + 1e-12)

        mean_tau = np.sum(delays * gains)
        var_tau = np.sum((delays - mean_tau) ** 2 * gains)
        return float(np.sqrt(var_tau) * 1e9)


class GeometricChannel(MultipathChannel):
    """Geometry-based stochastic channel model (GSCM).

    Extends MultipathChannel with:
    - Specified path positions
    - Configurable reflection points
    - Distance-based path loss

    Reference: 3GPP TR 38.901
    """

    def __init__(
        self,
        config: MultipathConfig,
        bs_position: tuple[float, float, float] = (0.0, 0.0, 25.0),
        uav_position: tuple[float, float, float] = (120.0, 45.0, 80.0),
        seed: int | None = None,
    ):
        """Initialize geometric channel.

        Args:
            config: Multipath channel configuration
            bs_position: Base station position
            uav_position: UAV position
            seed: Random seed
        """
        super().__init__(config, seed)
        self.bs_position = np.array(bs_position)
        self.uav_position = np.array(uav_position)

    def compute_path_loss_db(self, distance_m: float) -> float:
        """Compute path loss using 3GPP UMi model.

        Args:
            distance_m: Path distance in meters

        Returns:
            Path loss in dB
        """
        fc_ghz = self.config.carrier_frequency_hz / 1e9 if hasattr(self.config, 'carrier_frequency_hz') else 3.5

        if distance_m < 10:
            distance_m = 10.0

        pl_db = (
            32.0
            + 20.0 * np.log10(distance_m)
            + 20.0 * np.log10(fc_ghz)
        )
        return float(pl_db)


class StatisticalChannel(MultipathChannel):
    """Pure statistical channel model.

    Provides Rayleigh/Rician fading without geometry.
    Suitable for analytical studies and rapid prototyping.
    """

    def simulate_siso(
        self,
        num_samples: int,
        doppler_hz: float = 0.0,
    ) -> np.ndarray:
        """Simulate SISO fading channel.

        Args:
            num_samples: Number of time samples
            doppler_hz: Doppler frequency in Hz

        Returns:
            Complex channel coefficients [num_samples]
        """
        rng = self.rng

        if self.config.fading_type == FadingType.RAYLEIGH:
            h = (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)) / np.sqrt(2)

        elif self.config.fading_type == FadingType.RICIAN:
            k = 10.0 ** (self.config.rician_k_db / 10.0)
            los_amplitude = np.sqrt(k / (1 + k))
            multipath_amplitude = 1.0 / np.sqrt(1 + k)

            los_phase = rng.uniform(0, 2 * np.pi)
            mp_inphase = rng.standard_normal(num_samples) * multipath_amplitude / np.sqrt(2)
            mp_quadrature = rng.standard_normal(num_samples) * multipath_amplitude / np.sqrt(2)

            h = los_amplitude * np.exp(1j * los_phase) + mp_inphase + 1j * mp_quadrature

        if doppler_hz > 0:
            t = np.arange(num_samples) / num_samples
            doppler_filter = np.exp(-1j * 2 * np.pi * doppler_hz * t)
            h = h * doppler_filter

        return h


def generate_multipath_cir(
    config: MultipathConfig,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate multipath CIR from configuration (documented above)."""
    return MultipathChannel(config, seed=rng)._cir if False else (
        lambda cfg, r: _generate_cir_impl(cfg, r)
    )(config, rng)


def _generate_cir_impl(
    config: MultipathConfig,
    rng: np.random.Generator | None = None,
) -> dict:
    """Internal implementation of CIR generation."""
    if rng is None:
        rng = np.random.default_rng()

    num_elements = config.num_elements
    num_paths = config.num_paths

    delays = np.array(config.reflect_delays_ns[:num_paths]) * 1e-9
    gains_db = np.array(config.reflect_gains_db[:num_paths])
    gains_linear = 10.0 ** (gains_db / 20.0)
    doas_deg = np.array(config.reflect_doas_deg[:num_paths])

    a_mat = np.zeros((num_elements, num_paths), dtype=np.complex128)
    for p in range(num_paths):
        a_mat[:, p] = steering_vector_ula(
            num_elements, doas_deg[p], config.d_over_lambda
        )

    rician_k = 10.0 ** (config.rician_k_db / 10.0)

    if config.fading_type == FadingType.RICIAN and rician_k > 0.01:
        los_gain = np.sqrt(rician_k / (1 + rician_k))
        multipath_gain = 1.0 / np.sqrt(1 + rician_k)
        doas_deg = np.concatenate([[config.los_doa_deg], doas_deg])
        delays = np.concatenate([[0.0], delays])
        gains_linear = np.concatenate([[los_gain], gains_linear * multipath_gain])
        a_mat = np.zeros((num_elements, num_paths + 1), dtype=np.complex128)
        a_mat[:, 0] = steering_vector_ula(num_elements, config.los_doa_deg, config.d_over_lambda)
        for p in range(num_paths):
            a_mat[:, p + 1] = steering_vector_ula(
                num_elements, config.reflect_doas_deg[p], config.d_over_lambda
            )

    return {
        "delays": delays,
        "gains_linear": gains_linear,
        "doas_deg": doas_deg,
        "a_mat": a_mat,
        "num_paths": len(delays),
    }


def generate_multipath_cir(config, rng=None):
    """Wrapper for backward compatibility."""
    return _generate_cir_impl(config, rng)