"""Phase-2 DOA benchmark: classical + advanced surrogate methods."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from channel import UlaSceneConfig, simulate_ula_snapshots
from doa import (
    beamforming_estimate,
    doa_rmse_deg,
    esprit_estimate,
    music_estimate,
    ogfvbi_surrogate_estimate,
    transdoa_surrogate_estimate,
)


def _run_algorithms(
    cov: np.ndarray,
    num_elements: int,
    num_sources: int,
    angle_grid_deg: np.ndarray,
) -> list[tuple[str, np.ndarray, float]]:
    outputs: list[tuple[str, np.ndarray, float]] = []

    algo_funcs = [
        ("Beamforming", lambda: beamforming_estimate(cov, num_elements, num_sources, angle_grid_deg)),
        ("MUSIC", lambda: music_estimate(cov, num_elements, num_sources, angle_grid_deg)),
        ("ESPRIT", lambda: esprit_estimate(cov, num_sources)),
        (
            "TransDOA-surrogate",
            lambda: transdoa_surrogate_estimate(cov, num_elements, num_sources, angle_grid_deg),
        ),
        (
            "OGFVBI-surrogate",
            lambda: ogfvbi_surrogate_estimate(cov, num_elements, num_sources, angle_grid_deg),
        ),
    ]

    for name, fn in algo_funcs:
        t0 = time.perf_counter()
        try:
            est = np.asarray(fn(), dtype=float)
        except Exception:
            est = np.array([], dtype=float)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        outputs.append((name, est, dt_ms))

    return outputs


def _collect_records(
    sweep: str,
    snr_values: list[float],
    snapshot_values: list[int],
    trials: int,
    angle_grid_deg: np.ndarray,
) -> list[dict[str, float | str | int]]:
    records: list[dict[str, float | str | int]] = []

    target_doas = (-18.0, 21.0)
    num_elements = 8
    num_sources = len(target_doas)

    for scenario_name, multipath in [("LoS", False), ("NLoS", True)]:
        for snr_db in snr_values:
            for num_snapshots in snapshot_values:
                for trial in range(trials):
                    scene_cfg = UlaSceneConfig(
                        num_elements=num_elements,
                        num_snapshots=num_snapshots,
                        snr_db=snr_db,
                        target_doas_deg=target_doas,
                        multipath=multipath,
                    )
                    _, cov, true_doas = simulate_ula_snapshots(scene_cfg, seed=1000 + trial)

                    for algo_name, est, dt_ms in _run_algorithms(
                        cov,
                        num_elements=num_elements,
                        num_sources=num_sources,
                        angle_grid_deg=angle_grid_deg,
                    ):
                        rmse = doa_rmse_deg(est, true_doas)
                        records.append(
                            {
                                "sweep": sweep,
                                "scenario": scenario_name,
                                "snr_db": snr_db,
                                "num_snapshots": num_snapshots,
                                "trial": trial,
                                "algorithm": algo_name,
                                "rmse_deg": rmse,
                                "runtime_ms": dt_ms,
                            }
                        )

    return records


def _plot_rmse_vs_snr(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, scenario in zip(axes, ["LoS", "NLoS"]):
        df = summary[(summary["sweep"] == "snr") & (summary["scenario"] == scenario)]
        sns.lineplot(
            data=df,
            x="snr_db",
            y="rmse_mean",
            hue="algorithm",
            marker="o",
            ax=ax,
        )
        ax.set_title(f"RMSE vs SNR ({scenario})")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("RMSE (deg)")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].legend().remove()
    axes[1].legend().remove()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_rmse_vs_snapshots(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, scenario in zip(axes, ["LoS", "NLoS"]):
        df = summary[(summary["sweep"] == "snapshots") & (summary["scenario"] == scenario)]
        sns.lineplot(
            data=df,
            x="num_snapshots",
            y="rmse_mean",
            hue="algorithm",
            marker="o",
            ax=ax,
        )
        ax.set_title(f"RMSE vs Snapshots ({scenario})")
        ax.set_xlabel("Snapshots")
        ax.set_ylabel("RMSE (deg)")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].legend().remove()
    axes[1].legend().remove()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_runtime(summary: pd.DataFrame, out_path: Path) -> None:
    runtime = (
        summary.groupby("algorithm", as_index=False)["runtime_ms_mean"]
        .mean()
        .sort_values("runtime_ms_mean", ascending=False)
    )

    plt.figure(figsize=(8, 4.2))
    sns.barplot(data=runtime, x="algorithm", y="runtime_ms_mean")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Average runtime (ms)")
    plt.xlabel("")
    plt.title("Algorithm Runtime Comparison")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_scenario_adaptability(raw_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    focus = raw_df[(raw_df["snr_db"] == 5) & (raw_df["num_snapshots"] == 128)]
    agg = (
        focus.groupby(["algorithm", "scenario"], as_index=False)["rmse_deg"]
        .mean()
        .pivot(index="algorithm", columns="scenario", values="rmse_deg")
        .reset_index()
    )

    if "LoS" in agg.columns and "NLoS" in agg.columns:
        agg["nlos_los_ratio"] = agg["NLoS"] / np.maximum(agg["LoS"], 1e-12)
    else:
        agg["nlos_los_ratio"] = np.nan

    long_df = agg.melt(id_vars=["algorithm", "nlos_los_ratio"], value_vars=["LoS", "NLoS"], var_name="scenario", value_name="rmse_deg")

    plt.figure(figsize=(8.5, 4.2))
    sns.barplot(data=long_df, x="algorithm", y="rmse_deg", hue="scenario")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("RMSE at SNR=5dB, snapshots=128")
    plt.xlabel("")
    plt.title("LoS/NLoS Adaptability")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    return agg


def main() -> None:
    sns.set_theme(style="whitegrid")

    out_dir = PROJECT_ROOT / "results" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    angle_grid_deg = np.linspace(-80.0, 80.0, 321)

    # Sweep A: RMSE vs SNR (fixed snapshots)
    records_a = _collect_records(
        sweep="snr",
        snr_values=[-10, -5, 0, 5, 10, 15, 20],
        snapshot_values=[128],
        trials=12,
        angle_grid_deg=angle_grid_deg,
    )

    # Sweep B: RMSE vs snapshots (fixed SNR)
    records_b = _collect_records(
        sweep="snapshots",
        snr_values=[5],
        snapshot_values=[32, 64, 128, 256],
        trials=14,
        angle_grid_deg=angle_grid_deg,
    )

    raw_df = pd.DataFrame(records_a + records_b)
    raw_csv = out_dir / "phase2_benchmark_raw.csv"
    raw_df.to_csv(raw_csv, index=False)

    summary = (
        raw_df.groupby(["sweep", "scenario", "snr_db", "num_snapshots", "algorithm"], as_index=False)
        .agg(
            rmse_mean=("rmse_deg", "mean"),
            rmse_std=("rmse_deg", "std"),
            runtime_ms_mean=("runtime_ms", "mean"),
            runtime_ms_std=("runtime_ms", "std"),
        )
        .sort_values(["sweep", "scenario", "algorithm", "snr_db", "num_snapshots"])
    )
    summary_csv = out_dir / "phase2_benchmark_summary.csv"
    summary.to_csv(summary_csv, index=False)

    _plot_rmse_vs_snr(summary, out_dir / "phase2_rmse_vs_snr.png")
    _plot_rmse_vs_snapshots(summary, out_dir / "phase2_rmse_vs_snapshots.png")
    _plot_runtime(summary, out_dir / "phase2_runtime_comparison.png")

    adapt_df = _plot_scenario_adaptability(raw_df, out_dir / "phase2_los_nlos_adaptability.png")
    adapt_df.to_csv(out_dir / "phase2_scenario_adaptability.csv", index=False)

    print(f"Saved raw benchmark: {raw_csv}")
    print(f"Saved summary benchmark: {summary_csv}")
    print(f"Saved figures under: {out_dir}")


if __name__ == "__main__":
    main()
