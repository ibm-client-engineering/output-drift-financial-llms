"""Lightweight significance utilities for benchmark comparisons.

Permutation tests and correlation helpers for model-vs-model comparisons
on DFAH-Bench metrics.
"""

from typing import Callable, List, Tuple

import numpy as np
from scipy import stats


def permutation_test(
    metric_fn: Callable[[np.ndarray], float],
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = None,
) -> Tuple[float, float]:
    """Two-sample permutation test for difference in a metric.

    Args:
        metric_fn: Function that takes an array and returns a scalar.
        group_a: Observations from group A.
        group_b: Observations from group B.
        n_permutations: Number of random permutations.
        seed: Random seed for reproducibility.

    Returns:
        (observed_difference, two_sided_p_value)
    """
    a = np.asarray(group_a)
    b = np.asarray(group_b)
    rng = np.random.default_rng(seed)

    observed_diff = float(metric_fn(a) - metric_fn(b))

    combined = np.concatenate([a, b])
    n_a = len(a)
    count_extreme = 0

    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = metric_fn(combined[:n_a]) - metric_fn(combined[n_a:])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)
    return observed_diff, float(p_value)


def spearman_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Compute Spearman rank correlation with p-value.

    Thin wrapper around scipy.stats.spearmanr.

    Returns:
        (rho, p_value)
    """
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def compare_models(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn: Callable[[np.ndarray], float] = np.mean,
    n_permutations: int = 10000,
    seed: int = None,
) -> dict:
    """Compare two models on a metric with permutation test.

    Returns dict with observed_diff, p_value, mean_a, mean_b.
    """
    a = np.asarray(scores_a)
    b = np.asarray(scores_b)

    diff, p = permutation_test(metric_fn, a, b, n_permutations, seed)

    return {
        "observed_diff": diff,
        "p_value": p,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "n_a": len(a),
        "n_b": len(b),
    }
