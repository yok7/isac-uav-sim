"""Prepare DOA simulation figures for the first weekly report."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from doa.classical import (  # noqa: E402
    beamforming_spectrum,
    estimate_topk_angles,
    esprit_estimate,
    music_spectrum,
    steering_vector_ula,
)


OUT_DIR = PROJECT_ROOT / "report" / "weekly_report" / "assets"


def simulate_covariance(
    true_angles: list[float],
    num_elements: int = 8,
    num_snapshots: int = 512,
    snr_db: float = 8.0,
    seed: int = 13,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steering = np.stack(
        [steering_vector_ula(num_elements, angle) for angle in true_angles],
        axis=1,
    )
    sources = (
        rng.standard_normal((len(true_angles), num_snapshots))
        + 1j * rng.standard_normal((len(true_angles), num_snapshots))
    ) / np.sqrt(2)
    signal = steering @ sources

    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    snapshots = signal + noise
    return snapshots @ snapshots.conj().T / num_snapshots


def normalize_db(x: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(np.real(x) / np.max(np.real(x)), 1e-10))


def style_spectrum_axis(ax, title: str, true_angles: list[float], estimates: np.ndarray) -> None:
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Normalized spectrum (dB)")
    ax.set_xlim(-70, 70)
    ax.set_ylim(-45, 2)
    ax.grid(True, alpha=0.25)
    for angle in true_angles:
        ax.axvline(angle, color="black", linestyle="--", linewidth=1.1, alpha=0.65)
    for angle in estimates:
        ax.axvline(angle, color="#d62728", linestyle="-", linewidth=1.0, alpha=0.65)
    ax.text(
        0.02,
        0.95,
        "black dashed = true, red = estimated",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )


def plot_beamforming(cov: np.ndarray, grid: np.ndarray, true_angles: list[float]) -> None:
    spectrum = beamforming_spectrum(cov, num_elements=8, angle_grid_deg=grid)
    estimates = estimate_topk_angles(grid, spectrum, num_sources=2, min_separation_deg=8)

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.plot(grid, normalize_db(spectrum), color="#2b6cb0", linewidth=2.3)
    style_spectrum_axis(ax, "Conventional Beamforming: broad spatial peaks", true_angles, estimates)
    ax.text(
        0.02,
        0.08,
        f"estimated DOA: {', '.join(f'{x:.1f} deg' for x in estimates)}",
        transform=ax.transAxes,
        fontsize=10,
    )
    fig.savefig(OUT_DIR / "doa_beamforming_result.png", dpi=220)
    plt.close(fig)


def plot_music(cov: np.ndarray, grid: np.ndarray, true_angles: list[float]) -> None:
    spectrum = music_spectrum(cov, num_elements=8, num_sources=2, angle_grid_deg=grid)
    estimates = estimate_topk_angles(grid, spectrum, num_sources=2, min_separation_deg=8)

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.plot(grid, normalize_db(spectrum), color="#238b45", linewidth=2.3)
    style_spectrum_axis(ax, "MUSIC: sharp pseudo-spectrum peaks", true_angles, estimates)
    ax.text(
        0.02,
        0.08,
        f"estimated DOA: {', '.join(f'{x:.1f} deg' for x in estimates)}",
        transform=ax.transAxes,
        fontsize=10,
    )
    fig.savefig(OUT_DIR / "doa_music_result.png", dpi=220)
    plt.close(fig)


def plot_esprit(cov: np.ndarray, true_angles: list[float]) -> None:
    estimates = esprit_estimate(cov, num_sources=2)

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.set_title("ESPRIT: direct angle estimation without scanning", fontsize=15, pad=10)
    ax.set_xlim(-70, 70)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Angle (deg)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["estimated", "true"])
    ax.grid(True, axis="x", alpha=0.25)
    for angle in true_angles:
        ax.vlines(angle, 0.82, 1.18, color="black", linewidth=2.2)
        ax.scatter(angle, 1, color="black", s=80, zorder=3)
        ax.text(angle, 1.26, f"{angle:.0f}°", ha="center", fontsize=10)
    for angle in estimates:
        ax.vlines(angle, -0.18, 0.18, color="#d62728", linewidth=2.2)
        ax.scatter(angle, 0, color="#d62728", s=80, zorder=3)
        ax.text(angle, -0.36, f"{angle:.1f}°", ha="center", fontsize=10)
    ax.text(
        0.02,
        0.93,
        "uses shift invariance of two overlapping ULA subarrays",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )
    fig.savefig(OUT_DIR / "doa_esprit_result.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    true_angles = [-18.0, 24.0]
    cov = simulate_covariance(true_angles)
    grid = np.linspace(-70, 70, 1401)

    plot_beamforming(cov, grid, true_angles)
    plot_music(cov, grid, true_angles)
    plot_esprit(cov, true_angles)

    print("Saved DOA figures:")
    for name in [
        "doa_beamforming_result.png",
        "doa_music_result.png",
        "doa_esprit_result.png",
    ]:
        print(OUT_DIR / name)


if __name__ == "__main__":
    main()
