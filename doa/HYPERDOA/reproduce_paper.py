"""
HYPERDOA + Classical Methods Reproduction Script

Reproduces the paper experiments:
- HYPERDOA (lag feature, spatial smoothing)
- Classical subspace methods: MUSIC, Root-MUSIC, ESPRIT
- Varying SNR from -10 to 20 dB

Usage:
    python reproduce_paper.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SubspaceNet"))

import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import permutations
from datetime import datetime

from hyperdoa import HDCAoAModel, DOAConfig, evaluate_hdc, set_seed
from src.system_model import SystemModelParams, SystemModel
from src.signal_creation import Samples
from src.methods import MUSIC, RootMUSIC, Esprit


# ============================================================================
# Loss Function (same as SubspaceNet, consistent with HYPERDOA)
# ============================================================================

def MSPE_radians(predictions_rad, targets_rad):
    """Permutation-invariant MSPE in radians^2."""
    preds = np.atleast_2d(np.asarray(predictions_rad))
    tgts = np.atleast_2d(np.asarray(targets_rad))
    total, count = 0.0, 0
    for p, t in zip(preds, tgts):
        best = float("inf")
        for perm in permutations(p.tolist(), len(t)):
            err = (((np.array(perm) - t) + np.pi / 2) % np.pi) - np.pi / 2
            best = min(best, np.mean(err ** 2))
        total += best
        count += 1
    return total / max(count, 1)

def MSPE_db(predictions_rad, targets_rad):
    """MSPE in dB (same as HYPERDOA)."""
    mspe = max(MSPE_radians(predictions_rad, targets_rad), 1e-12)
    return 10 * np.log10(mspe)

def RMSPE_radians(predictions_rad, targets_rad):
    """Permutation-invariant RMSPE in radians."""
    preds = np.atleast_2d(np.asarray(predictions_rad))
    tgts = np.atleast_2d(np.asarray(targets_rad))
    total, count = 0.0, 0
    for p, t in zip(preds, tgts):
        best = float("inf")
        for perm in permutations(p.tolist(), len(t)):
            err = (((np.array(perm) - t) + np.pi / 2) % np.pi) - np.pi / 2
            best = min(best, np.sqrt(np.mean(err ** 2)))
        total += best
        count += 1
    return total / max(count, 1)

def RMSPE_degrees(predictions_deg, targets_deg):
    """Permutation-invariant RMSPE in degrees."""
    preds = np.atleast_2d(np.asarray(predictions_deg))
    tgts = np.atleast_2d(np.asarray(targets_deg))
    total, count = 0.0, 0
    for p, t in zip(preds, tgts):
        best = float("inf")
        for perm in permutations(p.tolist(), len(t)):
            err = (((np.array(perm) - t) + 90) % 180) - 90
            best = min(best, np.sqrt(np.mean(err ** 2)))
        total += best
        count += 1
    return total / max(count, 1)


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_dataset(N, M, T, snr, num_samples, signal_nature="non-coherent", seed=42):
    """Generate DOA dataset using SubspaceNet's signal generation."""
    params = SystemModelParams()
    params.N = N
    params.M = M
    params.T = T
    params.snr = snr
    params.signal_nature = signal_nature
    params.signal_type = "NarrowBand"
    params.eta = 0.0
    params.bias = 0.0
    params.sv_noise_var = 0.0

    np.random.seed(seed)
    torch.manual_seed(seed)

    samples_model = Samples(params)
    dataset = []
    for _ in range(num_samples):
        samples_model.set_doa(None)
        X_np, _, _, _ = samples_model.samples_creation(
            noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
        )
        X = torch.tensor(X_np, dtype=torch.complex64)
        Y = torch.tensor(samples_model.doa, dtype=torch.float64)
        dataset.append((X, Y))
    return dataset


# ============================================================================
# Classical Methods Evaluation
# ============================================================================

def evaluate_classical(dataset, N, M, T, snr, method_name):
    """Evaluate a classical subspace method on dataset."""
    params = SystemModelParams()
    params.N = N
    params.M = M
    params.T = T
    params.snr = snr
    params.signal_nature = "non-coherent"
    params.signal_type = "NarrowBand"
    params.eta = 0.0
    params.bias = 0.0
    params.sv_noise_var = 0.0

    samples_model = Samples(params)
    system_model = SystemModel(params)

    if method_name == "music":
        method = MUSIC(system_model)
    elif method_name == "root-music":
        method = RootMUSIC(system_model)
    elif method_name == "esprit":
        method = Esprit(system_model)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    preds_deg = []
    tgts_deg = []
    count = 0
    for X, Y in dataset:
        try:
            if method_name == "music":
                pred, _, M_est = method.narrowband(X=X, mode="sample")
            elif method_name == "root-music":
                pred, _, _, _, M_est = method.narrowband(X=X, mode="sample")
            elif method_name == "esprit":
                pred, M_est = method.narrowband(X=X, mode="sample")

            if pred.shape[0] < M:
                # Pad with random
                pad = np.round(np.random.rand(M - pred.shape[0]) * 180 - 90, 2)
                pred = np.concatenate([pred, pad])
            elif pred.shape[0] > M:
                pred = pred[:M]

            preds_deg.append(pred)
            tgts_deg.append(Y.cpu().numpy() * 180 / np.pi)
            count += 1
        except Exception:
            pass

    if count == 0:
        return float("nan")

    preds_deg = np.array(preds_deg)
    tgts_deg = np.array(tgts_deg)
    return RMSPE_degrees(preds_deg, tgts_deg)


# ============================================================================
# HYPERDOA Evaluation
# ============================================================================

def evaluate_hyperdoa(train_data, test_data, config, feature_type, seed=42):
    """Evaluate HYPERDOA on dataset."""
    set_seed(seed)
    loss, _ = evaluate_hdc(train_data, test_data, config, feature_type=feature_type,
                           return_model=True, verbose=False, seed=seed)
    return loss


# ============================================================================
# Main Reproduction
# ============================================================================

def main():
    print("=" * 70)
    print("  HYPERDOA Paper Reproduction")
    print("  ICASSP 2026: Hyperdimensional Computing for DOA Estimation")
    print("=" * 70)

    # Parameters (paper default)
    N = 8
    M = 3
    T = 100
    snr_list = [-10, -5, 0, 5, 10, 15, 20]
    methods = ["HYPERDOA-lag", "HYPERDOA-ss", "MUSIC", "Root-MUSIC", "ESPRIT"]

    results = {m: [] for m in methods}
    timing = {m: [] for m in methods}

    # Dataset sizes
    train_samples = 45000
    test_samples = 2250

    data_dir = Path("data_reproduced")
    data_dir.mkdir(exist_ok=True)

    print(f"\nParameters: N={N} sensors, M={M} sources, T={T} snapshots")
    print(f"SNR range: {snr_list}")
    print(f"Train: {train_samples}, Test: {test_samples}")
    print()

    for snr in snr_list:
        print("-" * 70)
        print(f"SNR = {snr:3d} dB")
        print("-" * 70)

        train_path = data_dir / f"train_N{N}_M{M}_T{T}_snr{snr}.pt"
        test_path = data_dir / f"test_N{N}_M{M}_T{T}_snr{snr}.pt"

        if train_path.exists() and test_path.exists():
            print(f"  Loading cached datasets...")
            train_data = torch.load(train_path, weights_only=False)
            test_data = torch.load(test_path, weights_only=False)
        else:
            print(f"  Generating datasets...")
            t0 = time.time()
            train_data = generate_dataset(N, M, T, snr, train_samples, seed=42)
            test_data = generate_dataset(N, M, T, snr, test_samples, seed=1142)
            torch.save(train_data, train_path)
            torch.save(test_data, test_path)
            print(f"  Generated in {time.time()-t0:.1f}s")

        config = DOAConfig(N=N, M=M, T=T, snr=float(snr))

        # HYPERDOA (lag)
        t0 = time.time()
        loss = evaluate_hyperdoa(train_data, test_data, config, "lag", seed=42)
        dt = time.time() - t0
        results["HYPERDOA-lag"].append(loss)
        timing["HYPERDOA-lag"].append(dt)
        print(f"  HYPERDOA-lag:  RMSPE = {loss:7.2f} dB  ({dt:5.1f}s)")

        # HYPERDOA (spatial smoothing)
        t0 = time.time()
        loss_ss = evaluate_hyperdoa(train_data, test_data, config, "spatial_smoothing", seed=42)
        dt = time.time() - t0
        results["HYPERDOA-ss"].append(loss_ss)
        timing["HYPERDOA-ss"].append(dt)
        print(f"  HYPERDOA-ss:   RMSPE = {loss_ss:7.2f} dB  ({dt:5.1f}s)")

        # Classical methods (evaluate on a subset for speed)
        subset_size = min(500, len(test_data))
        subset = test_data[:subset_size]

        for method in ["MUSIC", "Root-MUSIC", "ESPRIT"]:
            t0 = time.time()
            rmspe = evaluate_classical(subset, N, M, T, snr, method.lower())
            dt = time.time() - t0
            results[method].append(rmspe)
            timing[method].append(dt)
            print(f"  {method:<12s}  RMSPE = {rmspe:7.2f} deg  ({dt:5.1f}s)")

        print()

    # =========================================================================
    # Print Summary Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY (RMSPE in degrees)")
    print("=" * 70)
    print(f"{'Method':<15} " + " ".join(f"{snr:>8}" for snr in snr_list))
    print("-" * 70)
    for method in methods:
        vals = results[method]
        print(f"{method:<15} " + " ".join(f"{v:>8.2f}" for v in vals))
    print()

    # =========================================================================
    # Print Timing Table
    # =========================================================================
    print("=" * 70)
    print("  INFERENCE TIME (seconds per test set)")
    print("=" * 70)
    print(f"{'Method':<15} " + " ".join(f"{snr:>8}" for snr in snr_list))
    print("-" * 70)
    for method in methods:
        vals = timing[method]
        print(f"{method:<15} " + " ".join(f"{v:>8.2f}" for v in vals))
    print()

    # =========================================================================
    # Save Results
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = data_dir / f"results_{timestamp}.pt"
    torch.save({
        "snr_list": snr_list,
        "results": results,
        "timing": timing,
        "params": {"N": N, "M": M, "T": T, "train_samples": train_samples, "test_samples": test_samples}
    }, results_file)
    print(f"Results saved to {results_file}")

    # =========================================================================
    # Plot
    # =========================================================================
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 12})
    linestyles = {"HYPERDOA-lag": "-", "HYPERDOA-ss": "--", "MUSIC": "-.", "Root-MUSIC": ":", "ESPRIT": (0, (3, 1, 1, 1))}
    markers = {"HYPERDOA-lag": "o", "HYPERDOA-ss": "s", "MUSIC": "^", "Root-MUSIC": "D", "ESPRIT": "v"}
    colors = {"HYPERDOA-lag": "#1f77b4", "HYPERDOA-ss": "#2ca02c", "MUSIC": "#d62728", "Root-MUSIC": "#9467bd", "ESPRIT": "#ff7f0e"}

    for method in methods:
        plt.plot(snr_list, results[method], ls=linestyles[method], marker=markers[method],
                 color=colors[method], linewidth=2, markersize=6, label=method, alpha=0.85)

    plt.xlabel("SNR (dB)")
    plt.ylabel("RMSPE (degrees)")
    plt.title("DOA Estimation Performance vs SNR (N=8, M=3, T=100)")
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = data_dir / f"reproduction_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()

    return results, timing, snr_list


if __name__ == "__main__":
    results, timing, snr_list = main()
