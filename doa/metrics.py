"""Metrics for DOA experiments."""

from __future__ import annotations

import itertools

import numpy as np


def doa_rmse_deg(est_deg: np.ndarray, true_deg: np.ndarray) -> float:
    """Compute permutation-invariant RMSE in degrees."""
    est = np.asarray(est_deg, dtype=float)
    true = np.asarray(true_deg, dtype=float)

    if est.size == 0 or true.size == 0:
        return float("nan")

    k = min(est.size, true.size)
    est = np.sort(est)[:k]
    true = np.sort(true)[:k]

    best = float("inf")
    for perm in itertools.permutations(range(k)):
        err = est[list(perm)] - true
        rmse = float(np.sqrt(np.mean(err**2)))
        if rmse < best:
            best = rmse

    return best
