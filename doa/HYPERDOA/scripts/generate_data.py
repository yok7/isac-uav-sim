#!/usr/bin/env python
"""
Dataset Generation Script for HYPERDOA

This script generates synthetic DOA datasets using SubspaceNet's signal generation.
The generated datasets are directly compatible with HYPERDOA's expected format.

Requirements:
    - SubspaceNet repository cloned locally
    - Install: pip install tqdm

Usage:
    # First, clone SubspaceNet
    git clone https://github.com/ShlezingerLab/SubspaceNet.git

    # Then run this script
    python scripts/generate_data.py --subspacenet-path ../SubspaceNet --output data/

Example Settings:
    See --help for all available options.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm


def add_subspacenet_to_path(subspacenet_path: Path) -> bool:
    """Add SubspaceNet to Python path."""
    if not subspacenet_path.exists():
        return False
    sys.path.insert(0, str(subspacenet_path))
    return True


def generate_dataset(
    N: int,
    M: int,
    T: int,
    snr: float,
    num_samples: int,
    signal_nature: str = "non-coherent",
    signal_type: str = "NarrowBand",
    eta: float = 0.0,
    bias: float = 0.0,
    sv_noise_var: float = 0.0,
    seed: int = 42,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate DOA dataset using SubspaceNet's signal generation.

    Args:
        N: Number of sensors (array elements)
        M: Number of sources
        T: Number of time snapshots
        snr: Signal-to-noise ratio in dB
        num_samples: Number of samples to generate
        signal_nature: "non-coherent" or "coherent"
        signal_type: "NarrowBand" or "Broadband"
        eta: Location deviation (array imperfection)
        bias: Uniform spacing bias
        sv_noise_var: Steering vector noise variance
        seed: Random seed

    Returns:
        List of (X, Y) tuples where:
            X: Complex tensor of shape (N, T) - sensor observations
            Y: Tensor of shape (M,) - DOA angles in radians
    """
    # Import SubspaceNet modules
    try:
        from src.system_model import SystemModelParams
        from src.signal_creation import Samples
    except ImportError as e:
        raise ImportError(
            "SubspaceNet not found. Please clone it first:\n"
            "  git clone https://github.com/ShlezingerLab/SubspaceNet.git\n"
            "And provide the path with --subspacenet-path"
        ) from e

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create system model parameters
    params = SystemModelParams()
    params.N = N
    params.M = M
    params.T = T
    params.snr = snr
    params.signal_nature = signal_nature
    params.signal_type = signal_type
    params.eta = eta
    params.bias = bias
    params.sv_noise_var = sv_noise_var

    # Create samples generator
    samples_model = Samples(params)

    # Generate dataset
    dataset = []
    for _ in tqdm(range(num_samples), desc="Generating samples"):
        # Set random DOA (None triggers random generation with minimum gap)
        samples_model.set_doa(None)

        # Generate observations
        X_np, _, _, _ = samples_model.samples_creation(
            noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
        )

        # Convert to tensors
        X = torch.tensor(X_np, dtype=torch.complex64)
        Y = torch.tensor(samples_model.doa, dtype=torch.float64)

        dataset.append((X, Y))

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate DOA datasets for HYPERDOA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate datasets
    python scripts/generate_data.py --subspacenet-path ../SubspaceNet

    # Generate with specific SNR
    python scripts/generate_data.py --subspacenet-path ../SubspaceNet --snr 10

    # Generate coherent sources dataset
    python scripts/generate_data.py --subspacenet-path ../SubspaceNet --signal-nature coherent

Paper Experiment Configurations:
    Non-coherent, varying SNR:  --snr -10 to 20 (step 5)
    Coherent sources:           --signal-nature coherent
    Array imperfections:        --eta 0.1 --bias 0.05
        """,
    )

    # Paths
    parser.add_argument(
        "--subspacenet-path",
        type=Path,
        required=True,
        help="Path to cloned SubspaceNet repository",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output directory for datasets",
    )

    # System parameters
    parser.add_argument(
        "--N", type=int, default=8, help="Number of sensors (default: 8)"
    )
    parser.add_argument(
        "--M", type=int, default=3, help="Number of sources (default: 3)"
    )
    parser.add_argument(
        "--T", type=int, default=100, help="Number of snapshots (default: 100)"
    )
    parser.add_argument("--snr", type=float, default=-5, help="SNR in dB (default: -5)")

    # Signal parameters
    parser.add_argument(
        "--signal-nature",
        type=str,
        default="non-coherent",
        choices=["non-coherent", "coherent"],
        help="Signal nature (default: non-coherent)",
    )
    parser.add_argument(
        "--signal-type",
        type=str,
        default="NarrowBand",
        choices=["NarrowBand", "Broadband"],
        help="Signal type (default: NarrowBand)",
    )

    # Array imperfections
    parser.add_argument(
        "--eta", type=float, default=0.0, help="Location deviation (default: 0.0)"
    )
    parser.add_argument(
        "--bias", type=float, default=0.0, help="Uniform spacing bias (default: 0.0)"
    )
    parser.add_argument(
        "--sv-noise-var",
        type=float,
        default=0.0,
        help="Steering vector noise variance (default: 0.0)",
    )

    # Dataset sizes
    parser.add_argument(
        "--train-samples",
        type=int,
        default=45000,
        help="Number of training samples (default: 45000)",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=2250,
        help="Number of test samples (default: 2250)",
    )

    # Other
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Add SubspaceNet to path
    if not add_subspacenet_to_path(args.subspacenet_path):
        print(f"ERROR: SubspaceNet not found at {args.subspacenet_path}")
        print("Please clone it first:")
        print("  git clone https://github.com/ShlezingerLab/SubspaceNet.git")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print("HYPERDOA Dataset Generation")
    print("=" * 60)
    print(f"SubspaceNet path: {args.subspacenet_path}")
    print(f"Output directory: {args.output}")
    print()
    print("System Parameters:")
    print(f"  N (sensors):    {args.N}")
    print(f"  M (sources):    {args.M}")
    print(f"  T (snapshots):  {args.T}")
    print(f"  SNR:            {args.snr} dB")
    print(f"  Signal nature:  {args.signal_nature}")
    print(f"  Signal type:    {args.signal_type}")
    print()
    print("Array Imperfections:")
    print(f"  eta:            {args.eta}")
    print(f"  bias:           {args.bias}")
    print(f"  sv_noise_var:   {args.sv_noise_var}")
    print()
    print("Dataset Sizes:")
    print(f"  Train samples:  {args.train_samples}")
    print(f"  Test samples:   {args.test_samples}")
    print("=" * 60)

    # Generate training dataset
    print("\nGenerating training dataset...")
    train_data = generate_dataset(
        N=args.N,
        M=args.M,
        T=args.T,
        snr=args.snr,
        num_samples=args.train_samples,
        signal_nature=args.signal_nature,
        signal_type=args.signal_type,
        eta=args.eta,
        bias=args.bias,
        sv_noise_var=args.sv_noise_var,
        seed=args.seed,
    )

    # Generate test dataset (different seed)
    print("\nGenerating test dataset...")
    test_data = generate_dataset(
        N=args.N,
        M=args.M,
        T=args.T,
        snr=args.snr,
        num_samples=args.test_samples,
        signal_nature=args.signal_nature,
        signal_type=args.signal_type,
        eta=args.eta,
        bias=args.bias,
        sv_noise_var=args.sv_noise_var,
        seed=args.seed + 1000,  # Different seed for test
    )

    # Save datasets
    train_path = args.output / "train_dataset.pt"
    test_path = args.output / "test_dataset.pt"

    print(f"\nSaving training dataset to {train_path}...")
    torch.save(train_data, train_path)

    print(f"Saving test dataset to {test_path}...")
    torch.save(test_data, test_path)

    # Save metadata
    metadata = {
        "N": args.N,
        "M": args.M,
        "T": args.T,
        "snr": args.snr,
        "signal_nature": args.signal_nature,
        "signal_type": args.signal_type,
        "eta": args.eta,
        "bias": args.bias,
        "sv_noise_var": args.sv_noise_var,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "seed": args.seed,
    }
    metadata_path = args.output / "metadata.pt"
    torch.save(metadata, metadata_path)
    print(f"Saving metadata to {metadata_path}...")

    # Verify
    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"  Train: {train_path} ({len(train_data)} samples)")
    print(f"  Test:  {test_path} ({len(test_data)} samples)")
    print()
    print("Dataset format:")
    X, Y = train_data[0]
    print(f"  X shape: {X.shape} (complex64)")
    print(f"  Y shape: {Y.shape} (float64, radians)")
    print()
    print("Usage with HYPERDOA:")
    print("  from hyperdoa import evaluate_hdc, DOAConfig")
    print("  import torch")
    print()
    print("  train_data = torch.load('data/train_dataset.pt')")
    print("  test_data = torch.load('data/test_dataset.pt')")
    print(f"  config = DOAConfig(N={args.N}, M={args.M}, T={args.T})")
    print("  loss, model = evaluate_hdc(train_data, test_data, config)")


if __name__ == "__main__":
    main()
