"""Phase-2 communication-sensing tradeoff and Pareto experiments."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from channel import UlaSceneConfig, simulate_ula_snapshots
from doa import doa_rmse_deg, music_estimate
from waveform import IsacOfdmConfig, ber_proxy_qpsk, effective_comm_rate_bpshz, generate_isac_ofdm_frame


def _pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Pareto front for maximizing x and minimizing y."""
    ordered = df.sort_values(x_col, ascending=False).reset_index(drop=True)
    best_y = float("inf")
    picked = []

    for i, row in ordered.iterrows():
        y = float(row[y_col])
        if y <= best_y:
            best_y = y
            picked.append(i)

    return ordered.loc[picked].sort_values(x_col)


def _evaluate_one_setting(
    comm_ratio: float,
    pilot_density: float,
    scs_khz: float,
    snr_db: float,
    trials: int,
    angle_grid_deg: np.ndarray,
) -> dict[str, float]:
    cfg = IsacOfdmConfig(
        num_subcarriers=128,
        num_symbols=14,
        subcarrier_spacing_khz=scs_khz,
        pilot_density=pilot_density,
        comm_ratio=comm_ratio,
    )
    frame = generate_isac_ofdm_frame(cfg, seed=123)

    comm_fraction = float(frame["comm_fraction"])
    pilot_fraction = float(frame["pilot_fraction"])
    sensing_fraction = float(frame["sensing_fraction"])

    rate = effective_comm_rate_bpshz(
        snr_db=snr_db,
        comm_fraction=comm_fraction,
        pilot_fraction=pilot_fraction,
        modulation_order=cfg.modulation_order,
    )
    # Coarse waveform sensitivity factor for different subcarrier spacings.
    scs_rate_factor = {15.0: 0.97, 30.0: 1.00, 60.0: 1.04}.get(float(scs_khz), 1.0)
    rate *= scs_rate_factor
    ber = ber_proxy_qpsk(snr_db, pilot_fraction=pilot_fraction, sensing_fraction=sensing_fraction)

    # Map sensing resources to effective snapshots for DOA estimation quality.
    scs_sensing_factor = {15.0: 1.05, 30.0: 1.00, 60.0: 0.9}.get(float(scs_khz), 1.0)
    snapshots = int(np.clip(round((12 + 150 * sensing_fraction) * scs_sensing_factor), 12, 192))

    rmses = []
    for t in range(trials):
        # Effective sensing SNR degrades when communication ratio is high.
        sensing_snr_db = snr_db - 4.5 * comm_ratio
        scene_cfg = UlaSceneConfig(
            num_elements=8,
            num_snapshots=snapshots,
            snr_db=sensing_snr_db,
            target_doas_deg=(-18.3, 20.7),  # off-grid targets for a more realistic estimation setting
            multipath=True,
            multipath_atten_db=9.0,
        )
        _, cov, true_doas = simulate_ula_snapshots(scene_cfg, seed=7000 + t)
        est = music_estimate(cov, 8, 2, angle_grid_deg)
        rmses.append(doa_rmse_deg(est, true_doas))

    rmse = float(np.mean(rmses))

    return {
        "subcarrier_spacing_khz": scs_khz,
        "pilot_density": pilot_density,
        "comm_ratio": comm_ratio,
        "comm_fraction": comm_fraction,
        "pilot_fraction": pilot_fraction,
        "sensing_fraction": sensing_fraction,
        "snapshots_for_sensing": snapshots,
        "rate_bpshz": rate,
        "ber_proxy": ber,
        "rmse_deg": rmse,
    }


def _plot_pareto(default_df: pd.DataFrame, out_path: Path) -> None:
    front = _pareto_front(default_df, x_col="rate_bpshz", y_col="rmse_deg")

    plt.figure(figsize=(8.2, 5.2))
    sns.scatterplot(
        data=default_df,
        x="rate_bpshz",
        y="rmse_deg",
        hue="comm_ratio",
        palette="viridis",
        s=70,
    )
    plt.plot(front["rate_bpshz"], front["rmse_deg"], color="red", linewidth=2, label="Pareto front")
    plt.xlabel("Communication rate proxy (bps/Hz)")
    plt.ylabel("Sensing error (DOA RMSE, deg)")
    plt.title("Communication vs Sensing Tradeoff")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_ratio_curves(df: pd.DataFrame, out_path: Path) -> None:
    target = df[df["subcarrier_spacing_khz"] == 30.0].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    sns.lineplot(
        data=target,
        x="comm_ratio",
        y="rate_bpshz",
        hue="pilot_density",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Rate vs Communication Ratio")
    axes[0].set_xlabel("Communication resource ratio")
    axes[0].set_ylabel("Rate proxy (bps/Hz)")
    axes[0].grid(True, alpha=0.25)

    sns.lineplot(
        data=target,
        x="comm_ratio",
        y="rmse_deg",
        hue="pilot_density",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_title("RMSE vs Communication Ratio")
    axes[1].set_xlabel("Communication resource ratio")
    axes[1].set_ylabel("DOA RMSE (deg)")
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_ber_tradeoff(df: pd.DataFrame, out_path: Path) -> None:
    default = df[(df["subcarrier_spacing_khz"] == 30.0) & (df["pilot_density"] == 0.1)].copy()

    plt.figure(figsize=(7.8, 4.8))
    sns.lineplot(data=default, x="comm_ratio", y="ber_proxy", marker="o", label="BER proxy")
    sns.lineplot(data=default, x="comm_ratio", y="rmse_deg", marker="s", label="DOA RMSE (deg)")
    plt.yscale("log")
    plt.xlabel("Communication resource ratio")
    plt.ylabel("Log-scale metric")
    plt.title("Communication BER Proxy and Sensing RMSE")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _build_parameter_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scs, pilot), g in df.groupby(["subcarrier_spacing_khz", "pilot_density"]):
        g2 = g.copy()
        # Balanced utility: normalized(rate) - normalized(rmse)
        r = (g2["rate_bpshz"] - g2["rate_bpshz"].min()) / (g2["rate_bpshz"].max() - g2["rate_bpshz"].min() + 1e-12)
        e = (g2["rmse_deg"] - g2["rmse_deg"].min()) / (g2["rmse_deg"].max() - g2["rmse_deg"].min() + 1e-12)
        g2["utility"] = r - e
        best = g2.loc[g2["utility"].idxmax()]
        rows.append(
            {
                "subcarrier_spacing_khz": scs,
                "pilot_density": pilot,
                "recommended_comm_ratio": float(best["comm_ratio"]),
                "recommended_rate_bpshz": float(best["rate_bpshz"]),
                "recommended_rmse_deg": float(best["rmse_deg"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["subcarrier_spacing_khz", "pilot_density"])


def main() -> None:
    sns.set_theme(style="whitegrid")

    out_dir = PROJECT_ROOT / "results" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    angle_grid_deg = np.linspace(-80.0, 80.0, 321)

    records: list[dict[str, float]] = []
    for scs in [15.0, 30.0, 60.0]:
        for pilot_density in [0.05, 0.1, 0.2]:
            for comm_ratio in np.linspace(0.1, 0.9, 9):
                rec = _evaluate_one_setting(
                    comm_ratio=float(comm_ratio),
                    pilot_density=float(pilot_density),
                    scs_khz=float(scs),
                    snr_db=10.0,
                    trials=16,
                    angle_grid_deg=angle_grid_deg,
                )
                records.append(rec)

    df = pd.DataFrame(records)
    full_csv = out_dir / "phase2_tradeoff_full.csv"
    df.to_csv(full_csv, index=False)

    default_df = df[(df["subcarrier_spacing_khz"] == 30.0) & (df["pilot_density"] == 0.1)].copy()
    pareto_df = _pareto_front(default_df, x_col="rate_bpshz", y_col="rmse_deg")
    pareto_csv = out_dir / "phase2_tradeoff_pareto.csv"
    pareto_df.to_csv(pareto_csv, index=False)

    summary_df = _build_parameter_summary(df)
    summary_csv = out_dir / "phase2_tradeoff_parameter_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    _plot_pareto(default_df, out_dir / "phase2_pareto_rate_vs_rmse.png")
    _plot_ratio_curves(df, out_dir / "phase2_ratio_sweep_curves.png")
    _plot_ber_tradeoff(df, out_dir / "phase2_ber_rmse_tradeoff.png")

    print(f"Saved tradeoff full table: {full_csv}")
    print(f"Saved Pareto table: {pareto_csv}")
    print(f"Saved parameter summary: {summary_csv}")
    print(f"Saved figures under: {out_dir}")


if __name__ == "__main__":
    main()
