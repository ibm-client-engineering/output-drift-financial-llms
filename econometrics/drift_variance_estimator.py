"""
Drift Variance Estimation for Multi-Run LLM Output Analysis

Integrates output drift metrics with econometric measurement error framework.

Our Contribution:
    Standard econometric measurement error models assume:
        Y_llm = Y_true + epsilon_model (deterministic conditional on input)

    We decompose this into:
        Y_llm = Y_true + epsilon_model + epsilon_drift

    Where epsilon_drift captures RUN-TO-RUN variance for identical inputs.
    This drift variance component affects:
        1. Validation subsample sizing (Ludwig et al. 2024)
        2. Confidence interval width for debiased estimates
        3. Multi-run decision strategies (accept if k/n agree)

Practical Workflow:
    1. Run LLM k times per sample (k = 3-10)
    2. Estimate sigma_drift^2 per sample
    3. Flag high-drift samples for human review
    4. Use low-drift samples for validation debiasing
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter


@dataclass
class DriftAnalysis:
    """Results from multi-run drift analysis.

    Attributes:
        sample_ids: Sample identifiers
        drift_variances: Variance per sample (n,)
        drift_rates: Proportion of runs disagreeing with mode (n,)
        mean_drift_variance: Average drift variance
        high_drift_samples: Indices of high-drift samples
        low_drift_samples: Indices of low-drift samples
        threshold: Drift threshold used for flagging
    """
    sample_ids: List[str]
    drift_variances: np.ndarray
    drift_rates: np.ndarray
    mean_drift_variance: float
    high_drift_samples: List[int]
    low_drift_samples: List[int]
    threshold: float

    def summary(self) -> str:
        n = len(self.sample_ids)
        n_high = len(self.high_drift_samples)
        n_low = len(self.low_drift_samples)

        return f"""
Drift Variance Analysis
{'='*60}
Total samples:            {n}
Mean drift variance:      {self.mean_drift_variance:.3f}
Mean drift rate:          {self.drift_rates.mean():.1%}
---
High-drift samples:       {n_high} ({n_high/n:.1%}) → Human review
Low-drift samples:        {n_low} ({n_low/n:.1%}) → Validation pool
Threshold:                {self.threshold:.3f}

RECOMMENDATION:
  - Use {n_low} low-drift samples for validation debiasing
  - Route {n_high} high-drift samples to human review queue
  - Estimated validation subsample reduction: {n_low/n:.0%} of original size
"""


def estimate_sample_drift_variance(
    runs: np.ndarray,
    method: str = "variance"
) -> float:
    """Estimate drift variance for a single sample across k runs.

    Args:
        runs: Array of k runs for one sample, shape (k,)
        method: "variance", "iqr", "range"

    Returns:
        Drift variance estimate

    Example:
        >>> runs = np.array([1.0, 1.0, 1.0])  # perfect consistency
        >>> var = estimate_sample_drift_variance(runs)
        >>> print(var)  # 0.0
        >>>
        >>> runs = np.array([0.0, 1.0, 0.0])  # high drift
        >>> var = estimate_sample_drift_variance(runs)
        >>> print(var)  # 0.33 (2/3 disagree with mode)
    """
    k = len(runs)

    if method == "variance":
        # Sample variance (unbiased)
        return np.var(runs, ddof=1)

    elif method == "iqr":
        # Interquartile range (robust to outliers)
        q75, q25 = np.percentile(runs, [75, 25])
        iqr = q75 - q25
        # Convert IQR to variance estimate (for normal: sigma ≈ IQR / 1.349)
        return (iqr / 1.349) ** 2

    elif method == "range":
        # Range-based estimator
        range_val = np.max(runs) - np.min(runs)
        # d2 factors for small samples (n=3: 1.69, n=5: 2.33, n=10: 3.08)
        d2_factors = {3: 1.69, 4: 2.06, 5: 2.33, 10: 3.08}
        d2 = d2_factors.get(k, 2.33)
        return (range_val / d2) ** 2

    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_categorical_drift_rate(
    runs: np.ndarray
) -> Tuple[float, int]:
    """For categorical outputs, compute drift rate = proportion disagreeing with mode.

    Args:
        runs: Array of k runs (can be strings, ints, etc.)

    Returns:
        (drift_rate, mode_count)

    Example:
        >>> runs = np.array(['positive', 'positive', 'positive'])
        >>> drift_rate, mode_count = estimate_categorical_drift_rate(runs)
        >>> print(drift_rate)  # 0.0 (perfect agreement)
        >>>
        >>> runs = np.array(['positive', 'negative', 'positive'])
        >>> drift_rate, mode_count = estimate_categorical_drift_rate(runs)
        >>> print(drift_rate)  # 0.33 (1/3 disagree)
    """
    counts = Counter(runs)
    mode_value, mode_count = counts.most_common(1)[0]
    k = len(runs)
    drift_rate = (k - mode_count) / k
    return drift_rate, mode_count


def run_multi_run_analysis(
    sample_ids: List[str],
    runs_matrix: np.ndarray,
    categorical: bool = False,
    drift_threshold: float = 0.10
) -> DriftAnalysis:
    """Analyze drift across multiple samples with k runs each.

    Args:
        sample_ids: Sample identifiers (length n)
        runs_matrix: Shape (n, k) - k runs per sample
        categorical: If True, use categorical drift rate; else variance
        drift_threshold: Threshold for flagging high-drift samples

    Returns:
        DriftAnalysis object

    Example:
        >>> # Continuous outputs (e.g., sentiment scores)
        >>> runs = np.array([
        ...     [0.8, 0.8, 0.9],  # low drift
        ...     [0.5, -0.2, 0.7], # high drift
        ...     [1.0, 1.0, 1.0]   # zero drift
        ... ])
        >>> analysis = run_multi_run_analysis(['s1', 's2', 's3'], runs)
        >>> print(analysis.summary())
    """
    n, k = runs_matrix.shape

    drift_variances = np.zeros(n)
    drift_rates = np.zeros(n)

    for i in range(n):
        if categorical:
            drift_rate, _ = estimate_categorical_drift_rate(runs_matrix[i])
            drift_rates[i] = drift_rate
            drift_variances[i] = drift_rate  # for categorical, variance ~ drift rate
        else:
            drift_variances[i] = estimate_sample_drift_variance(runs_matrix[i])
            # For continuous, drift rate = variance / total_variance
            drift_rates[i] = min(drift_variances[i] / (1.0 + 1e-6), 1.0)

    mean_drift_variance = drift_variances.mean()

    # Flag high/low drift samples
    high_drift = [i for i in range(n) if drift_variances[i] > drift_threshold]
    low_drift = [i for i in range(n) if drift_variances[i] <= drift_threshold]

    return DriftAnalysis(
        sample_ids=sample_ids,
        drift_variances=drift_variances,
        drift_rates=drift_rates,
        mean_drift_variance=mean_drift_variance,
        high_drift_samples=high_drift,
        low_drift_samples=low_drift,
        threshold=drift_threshold
    )


def compute_optimal_k_runs(
    target_confidence: float = 0.95,
    expected_drift_rate: float = 0.10
) -> int:
    """Compute optimal number of runs k to achieve target confidence in drift detection.

    For categorical outputs with drift rate p:
        k runs needed for 95% CI width < epsilon:
        k = (1.96^2 * p * (1-p)) / epsilon^2

    Args:
        target_confidence: Desired confidence level (0.95 for 95%)
        expected_drift_rate: Expected proportion of drifting runs

    Returns:
        Recommended number of runs k

    Example:
        >>> k = compute_optimal_k_runs(0.95, 0.10)
        >>> print(k)  # ~5-7 runs needed
    """
    z = stats.norm.ppf((1 + target_confidence) / 2)  # 1.96 for 95%
    p = expected_drift_rate
    epsilon = 0.05  # tolerance

    k = int(np.ceil((z**2 * p * (1 - p)) / epsilon**2))
    return max(k, 3)  # minimum 3 runs


def majority_vote_decision(
    runs: np.ndarray,
    confidence_threshold: float = 0.60
) -> Tuple[any, float, bool]:
    """Make decision via majority vote across k runs.

    Args:
        runs: Array of k runs
        confidence_threshold: Minimum vote proportion to accept

    Returns:
        (decision, vote_proportion, is_confident)

    Example:
        >>> runs = np.array(['buy', 'buy', 'sell'])
        >>> decision, confidence, is_confident = majority_vote_decision(runs, 0.60)
        >>> print(decision)  # 'buy'
        >>> print(confidence)  # 0.67
        >>> print(is_confident)  # True (67% > 60% threshold)
    """
    counts = Counter(runs)
    decision, vote_count = counts.most_common(1)[0]
    k = len(runs)
    vote_proportion = vote_count / k
    is_confident = vote_proportion >= confidence_threshold

    return decision, vote_proportion, is_confident


def flag_for_human_review(
    drift_analysis: DriftAnalysis,
    review_threshold: float = 0.10
) -> pd.DataFrame:
    """Generate human review queue from high-drift samples.

    Args:
        drift_analysis: DriftAnalysis object
        review_threshold: Drift variance threshold for review

    Returns:
        DataFrame with samples flagged for review

    Example:
        >>> analysis = run_multi_run_analysis(sample_ids, runs_matrix)
        >>> review_queue = flag_for_human_review(analysis)
        >>> print(f"Review {len(review_queue)} samples")
    """
    review_rows = []

    for idx in drift_analysis.high_drift_samples:
        review_rows.append({
            'sample_id': drift_analysis.sample_ids[idx],
            'drift_variance': drift_analysis.drift_variances[idx],
            'drift_rate': drift_analysis.drift_rates[idx],
            'priority': 'HIGH' if drift_analysis.drift_variances[idx] > 2 * review_threshold else 'MEDIUM'
        })

    df = pd.DataFrame(review_rows)
    if len(df) > 0:
        df = df.sort_values('drift_variance', ascending=False)

    return df


def plot_drift_distribution(
    drift_analysis: DriftAnalysis,
    save_path: Optional[str] = None
):
    """Visualize drift variance distribution across samples.

    Args:
        drift_analysis: DriftAnalysis object
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of drift variances
    axes[0].hist(drift_analysis.drift_variances, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(drift_analysis.threshold, color='red', linestyle='--',
                    label=f'Threshold ({drift_analysis.threshold:.2f})')
    axes[0].set_xlabel('Drift Variance')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Drift Variances')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Drift rate histogram
    axes[1].hist(drift_analysis.drift_rates, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Drift Rate (Proportion Disagreeing)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Drift Rates')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


# ==============================================================================
# Example Usage: Headline Sentiment Multi-Run Analysis
# ==============================================================================

def example_headline_sentiment_drift_analysis():
    """
    Demonstrates drift variance estimation for sentiment labeling task.

    Task: Label headline sentiment (positive/negative/neutral)
    Model: Tier~1 (low drift) vs Tier~3 (high drift)
    Goal: Identify high-drift samples for human review
    """
    np.random.seed(42)
    n_samples = 100
    k_runs = 5

    print("="*60)
    print("Multi-Run Drift Analysis: Headline Sentiment Labeling")
    print("="*60)
    print()

    # Simulate Tier~1 model (low drift)
    print("TIER~1 MODEL (7-20B, 100% determinism at T=0.0):")
    true_labels = np.random.choice([-1, 0, 1], n_samples)  # ground truth
    tier1_runs = np.tile(true_labels.reshape(-1, 1), (1, k_runs))  # perfect consistency
    # Add tiny noise to ~5% of samples (represents edge cases)
    noise_idx = np.random.choice(n_samples, int(0.05 * n_samples), replace=False)
    for idx in noise_idx:
        tier1_runs[idx, np.random.randint(k_runs)] = np.random.choice([-1, 0, 1])

    tier1_analysis = run_multi_run_analysis(
        sample_ids=[f"headline_{i}" for i in range(n_samples)],
        runs_matrix=tier1_runs,
        categorical=True,
        drift_threshold=0.10
    )

    print(tier1_analysis.summary())
    print()

    # Simulate Tier~3 model (high drift)
    print("TIER~3 MODEL (120B, 12.5% determinism):")
    tier3_runs = np.random.choice([-1, 0, 1], (n_samples, k_runs))  # high variance
    # Force ~12.5% to be consistent (matches empirical findings)
    consistent_idx = np.random.choice(n_samples, int(0.125 * n_samples), replace=False)
    for idx in consistent_idx:
        tier3_runs[idx] = true_labels[idx]  # all runs agree

    tier3_analysis = run_multi_run_analysis(
        sample_ids=[f"headline_{i}" for i in range(n_samples)],
        runs_matrix=tier3_runs,
        categorical=True,
        drift_threshold=0.10
    )

    print(tier3_analysis.summary())
    print()

    # Comparison
    print("="*60)
    print("VALIDATION SUBSAMPLE IMPLICATIONS")
    print("="*60)
    print(f"Tier~1: {len(tier1_analysis.low_drift_samples)} samples usable for validation")
    print(f"        {len(tier1_analysis.high_drift_samples)} samples need human review")
    print()
    print(f"Tier~3: {len(tier3_analysis.low_drift_samples)} samples usable for validation")
    print(f"        {len(tier3_analysis.high_drift_samples)} samples need human review")
    print()

    validation_reduction = len(tier3_analysis.low_drift_samples) / len(tier1_analysis.low_drift_samples)
    print(f"KEY FINDING:")
    print(f"  Tier~3 provides {validation_reduction:.0%} of usable validation samples")
    print(f"  compared to Tier~1, requiring {1/validation_reduction:.1f}x larger")
    print(f"  initial dataset to achieve same validation power.")
    print()

    # Generate review queue
    tier3_review = flag_for_human_review(tier3_analysis)
    print(f"HUMAN REVIEW QUEUE (Tier~3 model):")
    print(tier3_review.head(10).to_string(index=False))
    print()

    # Optimal k runs
    k_optimal = compute_optimal_k_runs(0.95, 0.10)
    print(f"OPTIMAL NUMBER OF RUNS:")
    print(f"  For 95% confidence with 10% expected drift: k = {k_optimal} runs")
    print(f"  Current experiment uses k = {k_runs} runs")


if __name__ == "__main__":
    example_headline_sentiment_drift_analysis()
