"""ISAC DOA estimator bank: unified interface for MUSIC and HyperDOA."""

from __future__ import annotations

import time
import numpy as np

from doa.classical import steering_vector_ula

try:
    from doa import HyperDOAEstimator, HyperDOAConfig
except ImportError:
    HyperDOAEstimator = None
    HyperDOAConfig = None


C = 299_792_458.0


def estimate_doa_music_after_range(
    y: np.ndarray,
    x: np.ndarray,
    valid: np.ndarray,
    f_k: np.ndarray,
    range_est_m: float,
    tx_angle_est_deg: float,
    num_tx_ant: int,
    num_rx_ant: int,
    angle_grid_deg: np.ndarray,
) -> float:
    """Estimate DOA using MUSIC after range compensation.

    Args:
        y: Received signal [num_rx_ant, num_symbols, num_subcarriers]
        x: Transmitted signal [num_tx_ant, num_symbols, num_subcarriers]
        valid: Valid RE mask
        f_k: Subcarrier frequencies
        range_est_m: Estimated range for phase compensation
        tx_angle_est_deg: Estimated TX angle for steering
        num_tx_ant: Number of TX antennas
        num_rx_ant: Number of RX antennas
        angle_grid_deg: DOA search grid

    Returns:
        doa_est: Estimated DOA in degrees
    """
    a_tx = steering_vector_ula(num_tx_ant, tx_angle_est_deg) / np.sqrt(num_tx_ant)
    tx_field = np.einsum("t,tnk->nk", a_tx, x)

    # Range-induced phase compensation
    tau_hat = 2 * range_est_m / C
    h_range = np.exp(-1j * 2 * np.pi * f_k * tau_hat)

    # Collect snapshots from valid REs
    snapshots = []
    for sym in range(valid.shape[0]):
        for sc in range(valid.shape[1]):
            if valid[sym, sc] and abs(tx_field[sym, sc]) > 1e-8:
                denom = tx_field[sym, sc] * h_range[sc]
                snapshots.append(y[:, sym, sc] / denom)

    snapshots = np.asarray(snapshots, dtype=np.complex128)

    if snapshots.shape[0] < num_rx_ant:
        return float(angle_grid_deg[0])

    # Rx array covariance: [num_rx_ant, num_rx_ant]
    # snapshots shape: [num_snapshots, num_rx_ant]
    # ( snapshots.T @ snapshots.conj() ) -> [num_rx_ant, num_rx_ant]
    cov = snapshots.T @ snapshots.conj() / max(snapshots.shape[0], 1)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # One target: signal subspace dim = 1, rest is noise
    num_sources = 1
    noise_subspace = eigvecs[:, : num_rx_ant - num_sources]

    # MUSIC spectrum (noise subspace)
    spectrum = np.zeros_like(angle_grid_deg, dtype=float)
    for i, ang in enumerate(angle_grid_deg):
        a = steering_vector_ula(num_rx_ant, float(ang))
        denom = np.linalg.norm(noise_subspace.conj().T @ a) ** 2
        spectrum[i] = 1.0 / max(denom, 1e-12)

    return float(angle_grid_deg[np.argmax(spectrum)])


class DOAEstimatorBank:
    """Unified DOA estimator interface.

    Supports multiple estimators (MUSIC, HyperDOA) and returns their outputs
    via a common interface. Estimators are trained/initialized once at
    construction and reused across all trials.

    Example:
        >>> bank = DOAEstimatorBank(
        ...     algorithms=("music", "hyperdoa"),
        ...     num_rx_ant=8,
        ...     num_tx_ant=4,
        ...     num_snapshots=216,
        ...     device="cpu",
        ...     seed=42,
        ...     angle_grid_deg=np.linspace(-70, 70, 281),
        ... )
        >>> doa_deg = bank.estimate("music", y, x_tx, valid, f_k, range_est, angle_est)
        >>> doa_deg = bank.estimate("hyperdoa", y, x_tx, valid, f_k, range_est, angle_est)
    """

    def __init__(
        self,
        algorithms: tuple[str, ...],
        num_rx_ant: int,
        num_tx_ant: int,
        num_snapshots: int,
        device: str,
        seed: int,
        angle_grid_deg: np.ndarray,
    ):
        """Initialize DOA estimator bank.

        Args:
            algorithms: Tuple of estimator names ("music", "hyperdoa")
            num_rx_ant: Number of receive antennas
            num_tx_ant: Number of transmit antennas
            num_snapshots: Number of HDC snapshots (from valid RE count)
            device: "cuda" or "cpu"
            seed: Random seed for reproducibility
            angle_grid_deg: Search grid for MUSIC
        """
        self.algorithms = algorithms
        self.num_rx_ant = num_rx_ant
        self.num_tx_ant = num_tx_ant
        self.angle_grid_deg = angle_grid_deg

        self.hyperdoa: HyperDOAEstimator | None = None
        self.hyperdoa_training_ms: float | None = None

        if "hyperdoa" in algorithms:
            if HyperDOAEstimator is None or HyperDOAConfig is None:
                raise ImportError(
                    "HyperDOA is requested but not available. "
                    "Install dependencies: pip install torch-hd && cd doa/HYPERDOA && pip install -e ."
                )

            hypercfg = HyperDOAConfig(
                num_antennas=num_rx_ant,
                num_sources=1,
                num_snapshots=num_snapshots,
                n_dimensions=10000,
                min_angle_deg=-90.0,
                max_angle_deg=90.0,
                precision_deg=0.5,
                min_separation_deg=10.0,
                device=device,
                snr_db=10.0,
                training_samples=500,
            )

            self.hyperdoa = HyperDOAEstimator(hypercfg, seed=seed)

            t0 = time.perf_counter()
            self.hyperdoa.train()
            self.hyperdoa_training_ms = (time.perf_counter() - t0) * 1000.0

            print(
                f"[HyperDOA] Training done in {self.hyperdoa_training_ms:.0f} ms "
                f"(T={num_snapshots}, {hypercfg.training_samples} samples)."
            )

    def estimate(
        self,
        method: str,
        y: np.ndarray,
        x_tx: np.ndarray,
        valid: np.ndarray,
        f_k: np.ndarray,
        range_est_m: float,
        tx_angle_est_deg: float,
    ) -> float:
        """Estimate DOA using the specified method.

        Args:
            method: "music" or "hyperdoa"
            y: Received signal [num_rx_ant, num_symbols, num_subcarriers]
            x_tx: Transmitted signal [num_tx_ant, num_symbols, num_subcarriers]
            valid: Valid RE mask
            f_k: Subcarrier frequencies
            range_est_m: Estimated range for phase compensation
            tx_angle_est_deg: Estimated TX angle for steering

        Returns:
            doa_est_deg: Estimated DOA in degrees
        """
        if method == "hyperdoa":
            return self._estimate_hyperdoa(
                y=y,
                x_tx=x_tx,
                valid=valid,
                f_k=f_k,
                range_est_m=range_est_m,
                tx_angle_est_deg=tx_angle_est_deg,
            )

        if method == "music":
            return estimate_doa_music_after_range(
                y=y,
                x=x_tx,
                valid=valid,
                f_k=f_k,
                range_est_m=range_est_m,
                tx_angle_est_deg=tx_angle_est_deg,
                num_tx_ant=self.num_tx_ant,
                num_rx_ant=self.num_rx_ant,
                angle_grid_deg=self.angle_grid_deg,
            )

        raise ValueError(f"Unsupported DOA method: {method}")

    def _estimate_hyperdoa(
        self,
        y: np.ndarray,
        x_tx: np.ndarray,
        valid: np.ndarray,
        f_k: np.ndarray,
        range_est_m: float,
        tx_angle_est_deg: float,
    ) -> float:
        """Estimate DOA using HyperDOA after range compensation."""
        if self.hyperdoa is None:
            raise RuntimeError("HyperDOA was not initialized.")

        tau_hat = 2 * range_est_m / C
        h_range = np.exp(-1j * 2 * np.pi * f_k * tau_hat)

        a_tx = steering_vector_ula(self.num_tx_ant, tx_angle_est_deg)
        tx_field = np.einsum("t,tnk->nk", a_tx, x_tx)

        # Collect snapshots from valid REs
        snapshots = []
        for sym in range(valid.shape[0]):
            for sc in range(valid.shape[1]):
                if valid[sym, sc] and abs(tx_field[sym, sc]) > 1e-8:
                    denom = tx_field[sym, sc] * h_range[sc]
                    snapshots.append(y[:, sym, sc] / denom)

        snapshots = np.asarray(snapshots, dtype=np.complex128)

        if snapshots.shape[0] < self.num_rx_ant:
            return float(self.angle_grid_deg[0])

        # HyperDOA expects [N, T] — transpose [T, N] → [N, T]
        X_2d = snapshots.T  # [num_rx_ant, num_snapshots]

        # Power normalization to match HDC training signal level (~1-10 amplitude)
        power_scale = np.mean(np.abs(X_2d) ** 2)
        if power_scale > 0:
            X_2d = X_2d / np.sqrt(power_scale)

        return self.hyperdoa.predict_single(X_2d)