"""ISAC End-to-End Simulation — CLI entry point.

Orchestrates experiment tasks using Sionna RT ray-tracing and DOA estimators.
Reorganized into modular components:
  - end2end_config: SimulationConfig / SimulationResult
  - end2end_simulator: ISACSimulator
  - end2end_tasks: run_channel_comparison / run_multipath_study / run_moving_target_study
  - end2end_io: save_results / print_results_table / plot_results

Usage:
    python experiments/isac_end2end_simulation.py --mode full --num-trials 100
    python experiments/isac_end2end_simulation.py --compare-channels --num-trials 50
    python experiments/isac_end2end_simulation.py --multipath-study
    python experiments/isac_end2end_simulation.py --moving-target-study
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.end2end_config import SimulationConfig
from experiments.end2end_tasks import (
    run_channel_comparison,
    run_multipath_study,
    run_moving_target_study,
)
from experiments.end2end_io import (
    save_results,
    print_results_table,
    plot_results,
    plot_snapshot_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ISAC End-to-End Simulation with Sionna RT"
    )

    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "los", "nlos", "quick"])
    parser.add_argument("--snr-min", type=float, default=-5.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--snr-step", type=float, default=5.0)
    parser.add_argument("--num-trials", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--compare-channels", action="store_true")
    parser.add_argument("--multipath-study", action="store_true")
    parser.add_argument("--moving-target-study", action="store_true")

    parser.add_argument(
        "--doa-methods",
        nargs="+",
        default=["music"],
        choices=["music", "hyperdoa"],
        help="DOA estimators to evaluate (default: music)",
    )

    parser.add_argument(
        "--snapshots",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Snapshot counts for DOA estimators, e.g. --snapshots 16 32 64 128. "
            "If omitted, use all available snapshots."
        ),
    )

    parser.add_argument(
        "--snapshot-selection",
        type=str,
        default="first",
        choices=["first", "random"],
        help='How to select snapshots when limiting T: "first" or "random" (default: first)',
    )

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = SimulationConfig(
        mode=args.mode,
        snr_min_db=args.snr_min,
        snr_max_db=args.snr_max,
        snr_step_db=args.snr_step,
        num_trials=args.num_trials,
        seed=args.seed,
        algorithms=tuple(args.doa_methods),
        snapshot_counts=tuple(args.snapshots) if args.snapshots else (None,),
        snapshot_selection=args.snapshot_selection,
    )

    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    print("\nISAC End-to-End Simulation")
    print(f"Device: {device}")
    print(f"Target position: {config.uav_position} m")
    print(f"Output: {config.output_dir}")

    all_results = []

    if args.compare_channels:
        results = run_channel_comparison(config)
        all_results.extend(results)
        plot_results(results, config.output_dir)
        plot_snapshot_results(results, config.output_dir)

    if args.multipath_study:
        results = run_multipath_study(config)
        all_results.extend(results)

    if args.moving_target_study:
        results = run_moving_target_study(config)
        all_results.extend(results)

    if not args.compare_channels and not args.multipath_study and not args.moving_target_study:
        # Default: run channel comparison
        results = run_channel_comparison(config)
        all_results.extend(results)
        plot_results(results, config.output_dir)
        plot_snapshot_results(results, config.output_dir)

    if all_results:
        save_results(all_results, config.output_dir)
        print_results_table(all_results)

        print(f"\n{'=' * 60}")
        print(f"Simulation Complete")
        print(f"Total results: {len(all_results)}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()