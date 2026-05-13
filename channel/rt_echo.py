# channel/rt_echo.py
from __future__ import annotations

import numpy as np

from channel.rt_config import C


class RTEchoMixin:
    """ISAC echo synthesis on top of Sionna RT one-leg CFR."""

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
        if x_tx.ndim != 4:
            raise ValueError(
                "x_tx must have shape [batch, num_tx_ant, num_symbols, num_subcarriers], "
                f"got {x_tx.shape}"
            )

        batch_size, num_tx_ant, num_symbols, num_subcarriers = x_tx.shape

        bs_pos = self._bs_pos
        uav_pos = self._uav_pos

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
        )[0, :, :]

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
        )[:, 0, :]

        target_field = np.einsum("tk,btnk->bnk", h_out, x_tx)

        reflection_coeff = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))

        echo_signal = (
            reflection_coeff
            * h_ret[None, :, None, :]
            * target_field[:, None, :, :]
        )

        if target_velocity_mps is not None:
            rel = uav_pos - bs_pos
            unit_los = rel / (np.linalg.norm(rel) + 1e-12)
            radial_velocity = float(np.dot(target_velocity_mps, unit_los))

            wavelength = C / self.config.carrier_frequency_hz
            doppler_hz = 2.0 * radial_velocity / wavelength

            if ofdm_symbol_duration_s is None:
                ofdm_symbol_duration_s = 1.0 / subcarrier_spacing_hz

            t_sym = np.arange(num_symbols) * ofdm_symbol_duration_s
            doppler_phase = np.exp(1j * 2.0 * np.pi * doppler_hz * t_sym)
            echo_signal = echo_signal * doppler_phase[None, None, :, None]

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

        reflection_coeff = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))

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
            seed=self.seed,
        )[:, 0, :]

        target_field = np.einsum("tk,tnk->nk", h_out, x_tx)
        base_echo = reflection_coeff * h_ret[:, None, :] * target_field[None, :, :]

        wavelength = C / self.config.carrier_frequency_hz

        rel0 = uav_pos0 - bs_pos
        unit_los0 = rel0 / (np.linalg.norm(rel0) + 1e-12)
        radial_velocity = float(np.dot(vel, unit_los0))
        doppler_hz = 2.0 * radial_velocity / wavelength

        mask_3d = valid_mask[np.newaxis, :, :]

        for frame_idx in range(num_frames):
            if (
                frame_idx > 0
                and update_channel_every < num_frames
                and frame_idx % update_channel_every == 0
            ):
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

            t = frame_idx * frame_interval_s
            doppler_phase = np.exp(1j * 2.0 * np.pi * doppler_hz * t)

            signal = base_echo * doppler_phase

            selected_power = (
                np.sum(np.abs(signal) ** 2 * mask_3d)
                / np.count_nonzero(mask_3d)
            )
            noise_power = selected_power / (10 ** (snr_db / 10.0))

            noise = np.sqrt(noise_power / 2.0) * (
                rng.standard_normal(signal.shape)
                + 1j * rng.standard_normal(signal.shape)
            )

            y_echo_frames[frame_idx] = signal + noise

        return y_echo_frames