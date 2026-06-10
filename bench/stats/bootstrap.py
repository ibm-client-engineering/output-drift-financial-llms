"""Percentile bootstrap confidence intervals for DFAH-Bench metrics.

Simple, not overbuilt. Deterministic seed support required.

Example:
    >>> from bench.stats.bootstrap import bootstrap_ci
    >>> import numpy as np
    >>> data = np.array([0.8, 0.9, 0.85, 0.75, 0.95])
    >>> point, lo, hi = bootstrap_ci(np.mean, data, seed=42)
"""

from typing import Any, Callable, Tuple

import numpy as np


def bootstrap_ci(
    statistic_fn: Callable[[np.ndarray], float],
    data: np.ndarray,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = None,
) -> Tuple[float, float, float]:
    """Compute percentile bootstrap confidence interval.

    Args:
        statistic_fn: Function that takes an array and returns a scalar.
        data: 1-D array of observations.
        n_resamples: Number of bootstrap resamples.
        ci: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    data = np.asarray(data)
    rng = np.random.default_rng(seed)

    point_estimate = float(statistic_fn(data))

    n = len(data)
    bootstrap_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic_fn(sample)

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return point_estimate, ci_lower, ci_upper
