"""
Econometric Analysis of LLM Output Drift

A new research direction extending the LLM Output Drift findings into rigorous
econometric and agentic frameworks for financial AI.

This module builds on:
    1. LLM Output Drift paper findings (7-20B models, task-structure effects)
    2. Ludwig et al. (2024) econometric framework for LLMs
    3. Existing V2 harness infrastructure (harness/, metrics/, providers/)

Two Research Tracks:

    Track 1: Econometrics for LLM Measurement Error
        - Drift variance estimation
        - Semantic divergence metrics
        - Validation debiasing with drift correction
        - Training data leakage detection

    Track 2: Agentic AI Replayability (ICLR FinAI 2026)
        - Trajectory determinism metrics
        - Evidence-conditioned faithfulness
        - Drift stress-test harness
        - Determinism-faithfulness frontier

Usage:
    # Econometric track
    from econometrics.drift_variance_estimator import run_multi_run_analysis
    from econometrics.validation_debiasing import debiased_ols_regression

    # Agentic track
    from econometrics.agentic.metrics.trajectory_determinism import analyze_trajectory_determinism
    from econometrics.agentic.metrics.faithfulness import compute_faithfulness
    from econometrics.agentic.harness.stress_test_runner import StressTestHarness

Integration with Existing V2 Infrastructure:
    - Use `harness/load_models.py` for production load testing
    - Use `metrics/semantic_divergence_light.py` for PRSD-based metrics
    - Use `metrics/faithfulness.py` for 2x2 determinism matrix
    - Use `providers/watsonx.py` for cloud model testing

See README.md for research agenda and paper outlines.
"""

__version__ = "0.1.0"
__author__ = "AI4F Drift Research Team"

# Core econometric modules
from .drift_variance_estimator import (
    DriftAnalysis,
    run_multi_run_analysis,
    estimate_sample_drift_variance,
    compute_optimal_k_runs,
    majority_vote_decision
)

from .semantic_divergence_econometric import (
    SemanticDivergenceMetrics,
    SemanticEquivalenceClass,
    compute_semantic_divergence_metrics,
    detect_semantic_equivalence_classes,
    compute_effective_drift_rate_with_semantics
)

from .validation_debiasing import (
    ValidationSample,
    DebiasedEstimates,
    debiased_ols_regression,
    optimal_validation_size,
    compare_model_tiers_validation_requirements
)

from .leakage_detection import (
    LeakageReport,
    run_leakage_detection,
    detect_temporal_leakage
)

__all__ = [
    # Drift variance
    'DriftAnalysis',
    'run_multi_run_analysis',
    'estimate_sample_drift_variance',
    'compute_optimal_k_runs',
    'majority_vote_decision',
    # Semantic divergence
    'SemanticDivergenceMetrics',
    'SemanticEquivalenceClass',
    'compute_semantic_divergence_metrics',
    'detect_semantic_equivalence_classes',
    'compute_effective_drift_rate_with_semantics',
    # Validation debiasing
    'ValidationSample',
    'DebiasedEstimates',
    'debiased_ols_regression',
    'optimal_validation_size',
    'compare_model_tiers_validation_requirements',
    # Leakage detection
    'LeakageReport',
    'run_leakage_detection',
    'detect_temporal_leakage',
]
