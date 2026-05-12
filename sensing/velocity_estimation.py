"""Velocity estimation via slow-time Doppler FFT."""

from __future__ import annotations

import numpy as np


C = 299_792_458.0


def estimate_radial_velocity_doppler(
    y_frames: np.ndarray,
    valid_mask: np.ndarray,
    range_est_m: float,
    f_k: np.ndarray,
    frame_interval_s: float,
    carrier_frequency_hz: float,
) -> tuple[float, float]:
    """Estimate radial velocity via Doppler FFT across all subcarriers and antennas.

    Strategy: Compute Doppler FFT independently per (symbol, subcarrier)
    and average power spectra across valid REs and RX antennas. This preserves
    subcarrier and spatial diversity while maintaining slow-time coherence.

    Args:
        y_frames: Echo over frames [num_frames, num_rx_ant, num_symbols, num_subcarriers]
        valid_mask: Valid RE mask [num_symbols, num_subcarriers]
        range_est_m: Estimated range from range estimator (m)
        f_k: Subcarrier frequencies [num_subcarriers]
        frame_interval_s: Time between frames (seconds)
        carrier_frequency_hz: Carrier frequency in Hz

    Returns:
        est_velocity_mps: Estimated radial velocity (m/s)
        doppler_hz: Estimated Doppler frequency (Hz)
    """
    num_frames = y_frames.shape[0]
    num_rx_ant = y_frames.shape[1]
    num_symbols = y_frames.shape[2]
    num_subcarriers = y_frames.shape[3]

    # Phase compensation per subcarrier to remove range-induced phase
    tau_est = 2 * range_est_m / C
    phase_comp = np.exp(-1j * 2 * np.pi * tau_est * f_k)  # [num_subcarriers]

    # Accumulate Doppler power spectra: sum |FFT(slow_time)|^2 over all RE × RX
    accumulated_spectrum = np.zeros(num_frames, dtype=float)

    for sym in range(num_symbols):
        for sc in range(num_subcarriers):
            if not valid_mask[sym, sc]:
                continue

            # Collect slow-time vector for this RE across frames
            # Shape: [num_frames, num_rx_ant]
            slow_time_matrix = y_frames[:, :, sym, sc] * phase_comp[sc]

            # Apply Hann window to reduce spectral leakage
            window = np.hanning(num_frames)[:, None]
            fft_matrix = np.fft.fft(slow_time_matrix * window, n=num_frames, axis=0)
            power_spectrum = np.abs(fft_matrix) ** 2  # [num_frames, num_rx_ant]

            # Sum over RX antennas → one spectrum per RE
            accumulated_spectrum += np.sum(power_spectrum, axis=1)

    # Find Doppler peak using fftshift for symmetric spectrum
    freqs = np.fft.fftshift(np.fft.fftfreq(num_frames, d=frame_interval_s))
    spectrum_shifted = np.fft.fftshift(accumulated_spectrum)

    peak_idx = int(np.argmax(spectrum_shifted))
    peak_val = spectrum_shifted[peak_idx]

    # Parabolic interpolation for sub-bin accuracy
    if 0 < peak_idx < num_frames - 1:
        alpha = spectrum_shifted[peak_idx - 1]
        beta = peak_val
        gamma = spectrum_shifted[peak_idx + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-12:
            delta = 0.5 * (alpha - gamma) / denom
        else:
            delta = 0.0
        doppler_hz = freqs[peak_idx] + delta * (freqs[1] - freqs[0])
    else:
        doppler_hz = freqs[peak_idx]

    wavelength = C / carrier_frequency_hz
    est_velocity_mps = doppler_hz * wavelength / 2.0

    return est_velocity_mps, doppler_hz