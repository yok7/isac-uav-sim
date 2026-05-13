"""I/O utilities: CSV saving, result table printing, plotting."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.end2end_config import SimulationResult


def save_results(results: list[SimulationResult], output_dir: Path) -> None:
    """Save results to CSV.

    Args:
        results: List of SimulationResult objects
        output_dir: Output directory
    """
    output_path = output_dir / "simulation_results.csv"

    fieldnames = [
        "channel_type", "doa_method", "max_depth",
        "specular_reflection", "diffuse_scattering",
        "target_velocity_mps", "true_radial_velocity_mps", "snr_db",
        "num_snapshots", "available_snapshots",
        "range_rmse_m", "range_bias_m", "range_std_m",
        "doa_rmse_deg", "doa_bias_deg", "doa_std_deg",
        "vel_rmse_mps", "vel_bias_mps", "vel_std_mps",
        "runtime_ms", "num_trials",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "channel_type": result.channel_type,
                "doa_method": result.doa_method,
                "max_depth": result.max_depth,
                "specular_reflection": result.specular_reflection,
                "diffuse_scattering": result.diffuse_scattering,
                "target_velocity_mps": f"{result.target_velocity_mps:.2f}",
                "snr_db": f"{result.snr_db:.1f}",
                "num_snapshots": result.num_snapshots if result.num_snapshots is not None else "",
                "available_snapshots": result.available_snapshots if result.available_snapshots is not None else "",
                "range_rmse_m": f"{result.range_rmse_m:.4f}",
                "range_bias_m": f"{result.range_bias_m:.4f}",
                "range_std_m": f"{result.range_std_m:.4f}",
                "doa_rmse_deg": f"{result.doa_rmse_deg:.4f}",
                "doa_bias_deg": f"{result.doa_bias_deg:.4f}",
                "doa_std_deg": f"{result.doa_std_deg:.4f}",
                "runtime_ms": f"{result.runtime_ms:.2f}",
                "num_trials": result.num_trials,
            }
            if result.vel_rmse_mps is not None:
                row["vel_rmse_mps"] = f"{result.vel_rmse_mps:.4f}"
                row["vel_bias_mps"] = f"{result.vel_bias_mps:.4f}"
                row["vel_std_mps"] = f"{result.vel_std_mps:.4f}"
                row["true_radial_velocity_mps"] = (
                    f"{result.true_radial_velocity_mps:.4f}"
                    if result.true_radial_velocity_mps is not None
                    else ""
                )
            else:
                row["vel_rmse_mps"] = ""
                row["vel_bias_mps"] = ""
                row["vel_std_mps"] = ""
                row["true_radial_velocity_mps"] = ""
            writer.writerow(row)

    print(f"\nResults saved to: {output_path}")


def print_results_table(results: list[SimulationResult]) -> None:
    """Print a formatted table of results.

    Args:
        results: List of SimulationResult objects
    """
    print("\n" + "=" * 165)
    print(
        f"{'Channel':<22} {'DOA':<10} {'MaxD':>5} {'Spec':>5} {'Diff':>5} "
        f"{'Vel':>6} {'TrueRadial':>10} {'SNR':>6} {'T':>6} "
        f"{'RangeRMSE':>10} {'DOARMSE':>10} {'VelRMSE':>10} {'VelStd':>8}"
    )
    print("-" * 165)

    for r in results:
        vel_str = f"{r.vel_rmse_mps:.2f}" if r.vel_rmse_mps is not None else "      N/A"
        vel_std_str = (
            f"{r.vel_std_mps:.2f}" if r.vel_std_mps is not None else "     N/A"
        )
        true_rad_str = (
            f"{r.true_radial_velocity_mps:.2f}"
            if r.true_radial_velocity_mps is not None
            else "        N/A"
        )
        print(
            f"{r.channel_type:<22} {r.doa_method:<10} {r.max_depth:>5} "
            f"{str(r.specular_reflection):>5} {str(r.diffuse_scattering):>5} "
            f"{r.target_velocity_mps:>6.1f} {true_rad_str:>10} {r.snr_db:>6.1f} "
            f"{str(r.num_snapshots) if r.num_snapshots is not None else 'all':>6} "
            f"{r.range_rmse_m:>10.3f} {r.doa_rmse_deg:>10.3f} "
            f"{vel_str:>10} {vel_std_str:>8}"
        )

    print("=" * 165)


def plot_results(results: list[SimulationResult], output_dir: Path) -> None:
    """Plot DOA and range RMSE vs SNR.

    Generates a two-panel figure:
    - Left: DOA RMSE vs SNR
    - Right: Range RMSE vs SNR

    Args:
        results: List of SimulationResult objects
        output_dir: Output directory
    """
    channel_types = sorted(set(r.channel_type for r in results))
    doa_methods = sorted(set(r.doa_method for r in results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ch_colors = {
        "LOS": "#3182bd",
        "NLoS": "#31a354",
        "StreetCanyon": "#e6550d",
        "LoS only": "#6baed6",
        "LoS + specular, depth=1": "#74c476",
        "LoS + specular, depth=2": "#31a354",
        "Full multipath, depth=3": "#e6550d",
        "Static": "#3182bd",
        "10 m/s (rx)": "#31a354",
        "20 m/s (rx)": "#e6550d",
        "10 m/s (tx)": "#756bb1",
        "los": "#3182bd",
        "nlos": "#31a354",
        "street_canyon": "#e6550d",
    }
    ch_markers = {
        "LOS": "o",
        "NLoS": "s",
        "StreetCanyon": "^",
        "LoS only": "o",
        "LoS + specular, depth=1": "s",
        "LoS + specular, depth=2": "^",
        "Full multipath, depth=3": "D",
        "Static": "o",
        "10 m/s (rx)": "s",
        "20 m/s (rx)": "^",
        "10 m/s (tx)": "D",
        "los": "o",
        "nlos": "s",
        "street_canyon": "^",
    }
    method_linestyles = {
        "music": "-",
        "hyperdoa": ":",
        "esprit": "-.",
    }

    for ch_type in channel_types:
        for method in doa_methods:
            ch_results = [
                r for r in results
                if r.channel_type == ch_type and r.doa_method == method
            ]
            if not ch_results:
                continue
            snrs = [r.snr_db for r in ch_results]
            rmses = [r.doa_rmse_deg for r in ch_results]
            label = f"{ch_type} [{method}]"
            ls = method_linestyles.get(method, "-")
            axes[0].plot(
                snrs, rmses,
                marker=ch_markers.get(ch_type, "o"),
                color=ch_colors.get(ch_type, "gray"),
                label=label,
                linewidth=1.5,
                markersize=7,
                linestyle=ls,
            )

    axes[0].set_xlabel("SNR (dB)", fontsize=12)
    axes[0].set_ylabel("DOA RMSE (deg)", fontsize=12)
    axes[0].set_title("DOA Estimation vs SNR", fontsize=14)
    axes[0].legend(fontsize=8, loc="best")
    axes[0].grid(True, alpha=0.3)

    for ch_type in channel_types:
        for method in doa_methods:
            ch_results = [
                r for r in results
                if r.channel_type == ch_type and r.doa_method == method
            ]
            if not ch_results:
                continue
            snrs = [r.snr_db for r in ch_results]
            rmses = [r.range_rmse_m for r in ch_results]
            label = f"{ch_type} [{method}]"
            ls = method_linestyles.get(method, "-")
            axes[1].plot(
                snrs, rmses,
                marker=ch_markers.get(ch_type, "o"),
                color=ch_colors.get(ch_type, "gray"),
                label=label,
                linewidth=1.5,
                markersize=7,
                linestyle=ls,
            )

    axes[1].set_xlabel("SNR (dB)", fontsize=12)
    axes[1].set_ylabel("Range RMSE (m)", fontsize=12)
    axes[1].set_title("Range Estimation vs SNR", fontsize=14)
    axes[1].legend(fontsize=8, loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "end2end_rmse_vs_snr.png", dpi=180)
    plt.close(fig)