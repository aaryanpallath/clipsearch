"""Retrieval metrics and the statistics to compare systems honestly.

Each image has about five captions, so queries are not independent: confidence intervals resample *images* (with
all their captions), a cluster bootstrap. Comparisons are paired: both systems are scored on the same resamples."""

from __future__ import annotations

import numpy as np


def ranks(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    """1-based rank of each query's correct image. Ties count against the system (pessimistic)."""
    correct = scores[np.arange(len(target)), target]
    return (scores >= correct[:, None]).sum(axis=1)


def recall_at(r: np.ndarray, k: int) -> float:
    return float((r <= k).mean())


def summary(r: np.ndarray) -> dict[str, float]:
    return {
        "R@1": recall_at(r, 1),
        "R@5": recall_at(r, 5),
        "R@10": recall_at(r, 10),
        "MRR": float((1.0 / r).mean()),
        "median_rank": float(np.median(r)),
    }


def _clusters(groups: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(groups, kind="stable")
    _, starts = np.unique(groups[order], return_index=True)
    return np.split(order, starts[1:])


def bootstrap_ci(
    values: np.ndarray, groups: np.ndarray, n: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean of per-query `values` with a cluster-bootstrap CI (clusters = images)."""
    clusters = _clusters(groups)
    sums = np.array([values[c].sum() for c in clusters])
    counts = np.array([len(c) for c in clusters])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(clusters), size=(n, len(clusters)))
    stats = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(stats, [(1 - level) / 2, (1 + level) / 2])
    return float(values.mean()), float(lo), float(hi)


def paired_difference(a: np.ndarray, b: np.ndarray, groups: np.ndarray, **kw) -> dict[str, float]:
    """Mean of b - a with a paired cluster-bootstrap CI."""
    diff = b.astype(float) - a.astype(float)
    mean, lo, hi = bootstrap_ci(diff, groups, **kw)
    return {"diff": mean, "ci_low": lo, "ci_high": hi}
