"""Classical DOA demo: Beamforming vs MUSIC on a single ULA target."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from doa.classical import (
    beamforming_spectrum,
    estimate_peak_angle,
    music_spectrum,
    steering_vector_ula,
)


def simulate_snapshots(
    num_elements: int = 8,
    true_angle_deg: float = 20.0,
    snr_db: float = 5.0,
    num_snapshots: int = 256,
) -> np.ndarray:
    a = steering_vector_ula(num_elements, true_angle_deg)
    s = (np.random.randn(num_snapshots) + 1j * np.random.randn(num_snapshots)) / np.sqrt(2)
    # Snapshot model x[t] = a(theta) * s[t]  (column-vector convention)
    x_signal = np.outer(s, np.conj(a))

    signal_power = np.mean(np.abs(x_signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(num_snapshots, num_elements)
        + 1j * np.random.randn(num_snapshots, num_elements)
    )
    x = x_signal + noise
    cov = (x.conj().T @ x) / num_snapshots
    return cov


def main():
    np.random.seed(7)

    num_elements = 8
    true_angle = 20.0
    cov = simulate_snapshots(
        num_elements=num_elements,
        true_angle_deg=true_angle,
        snr_db=5.0,
        num_snapshots=256,
    )

    grid = np.linspace(-90, 90, 721)
    p_bf = beamforming_spectrum(cov, num_elements, grid)
    p_music = music_spectrum(cov, num_elements, num_sources=1, angle_grid_deg=grid)

    p_bf_db = 10 * np.log10(np.maximum(p_bf / np.max(p_bf), 1e-10))
    p_music_db = 10 * np.log10(np.maximum(p_music / np.max(p_music), 1e-10))

    est_bf = estimate_peak_angle(grid, p_bf)
    est_music = estimate_peak_angle(grid, p_music)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "week2_doa_demo_beamforming_music.png"

    plt.figure(figsize=(8, 4.8))
    plt.plot(grid, p_bf_db, label=f"Beamforming (est={est_bf:.1f} deg)", linewidth=2)
    plt.plot(grid, p_music_db, label=f"MUSIC (est={est_music:.1f} deg)", linewidth=2)
    plt.axvline(true_angle, color="k", linestyle="--", alpha=0.6, label=f"True angle={true_angle:.1f} deg")
    plt.ylim([-50, 2])
    plt.xlim([-90, 90])
    plt.grid(True, alpha=0.25)
    plt.xlabel("Angle (deg)")
    plt.ylabel("Normalized spectrum (dB)")
    plt.title("DOA Demo: Beamforming vs MUSIC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)

    print(f"Saved: {out_path.resolve()}")
    print(f"True angle: {true_angle:.1f} deg")
    print(f"Beamforming estimate: {est_bf:.1f} deg")
    print(f"MUSIC estimate: {est_music:.1f} deg")


if __name__ == "__main__":
    main()
