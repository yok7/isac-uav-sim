"""Week-1 starter task: plot ULA array response pattern for 8 elements."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ula_response(num_elements: int = 8, d_over_lambda: float = 0.5):
    angles_deg = np.linspace(-90, 90, 721)
    theta = np.deg2rad(angles_deg)

    n = np.arange(num_elements)
    steering = np.exp(1j * 2 * np.pi * d_over_lambda * np.outer(np.sin(theta), n))
    weights = np.ones(num_elements) / num_elements  # uniform weights

    af = steering @ weights
    af_mag = np.abs(af)
    af_db = 20 * np.log10(np.maximum(af_mag / np.max(af_mag), 1e-8))
    return angles_deg, af_db


def main():
    angles, af_db = ula_response(num_elements=8, d_over_lambda=0.5)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "week1_ula_pattern.png"

    plt.figure(figsize=(8, 4.5))
    plt.plot(angles, af_db, linewidth=2)
    plt.ylim([-40, 1])
    plt.xlim([-90, 90])
    plt.grid(True, alpha=0.3)
    plt.xlabel("Angle (deg)")
    plt.ylabel("Normalized response (dB)")
    plt.title("8-element ULA Array Response")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
