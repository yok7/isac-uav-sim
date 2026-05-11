"""
Basic training example for HYPERDOA.

This example demonstrates:
1. Loading a pre-generated dataset
2. Training an HDC model
3. Evaluating on test data
4. Saving the trained model

Dataset format:
    List of (X, Y) tuples where:
    - X: Complex tensor of shape (N, T) - sensor observations
    - Y: Tensor of shape (M,) - ground truth DOA in radians

Usage:
    python examples/train_basic.py --data-dir data/

Note:
    You need to generate datasets first using SubspaceNet or similar tools.
    See README.md for dataset generation instructions.
"""

import argparse
from pathlib import Path
import torch

# Add parent directory to path for local development
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hyperdoa import HDCAoAModel, DOAConfig, evaluate_hdc, set_seed, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train HDC model for DOA estimation")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Data directory"
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="lag",
        choices=["lag", "spatial_smoothing"],
        help="Feature extraction method",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--N", type=int, default=8, help="Number of sensors")
    parser.add_argument("--M", type=int, default=2, help="Number of sources")
    parser.add_argument("--T", type=int, default=100, help="Number of snapshots")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Configure system parameters
    config = DOAConfig(
        N=args.N,
        M=args.M,
        T=args.T,
    )

    print("=" * 60)
    print("HYPERDOA - Basic Training Example")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  N (sensors): {config.N}")
    print(f"  M (sources): {config.M}")
    print(f"  T (snapshots): {config.T}")
    print(f"  Feature type: {args.feature_type}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    # Load datasets
    train_path = args.data_dir / "train_dataset.pt"
    test_path = args.data_dir / "test_dataset.pt"

    if not train_path.exists() or not test_path.exists():
        print("\nERROR: Dataset files not found!")
        print(f"  Expected: {train_path}")
        print(f"  Expected: {test_path}")
        print("\nTo generate datasets, use SubspaceNet:")
        print("  https://github.com/ShlezingerLab/SubspaceNet")
        print("\nDataset format: List of (X, Y) tuples")
        print("  X: Complex tensor (N, T)")
        print("  Y: Tensor (M,) in radians")
        return

    print("\nLoading datasets...")
    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)

    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")

    # Train and evaluate
    print("\nTraining...")
    test_loss, model = evaluate_hdc(
        train_data=train_data,
        test_data=test_data,
        config=config,
        feature_type=args.feature_type,
        return_model=True,
        verbose=True,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print(f"Final Test MSPE: {test_loss:.2f} dB")
    print("=" * 60)

    # Save model
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"hdc_{args.feature_type}_model.pt"

    save_checkpoint(model, str(ckpt_path), meta={"feature_type": args.feature_type})
    print(f"\nModel saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
