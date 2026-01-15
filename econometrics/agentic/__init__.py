"""
Agentic AI Replayability Framework

ICLR 2026 FinAI Workshop Research Track:
    "Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness
     for Tool-Using LLM Agents"

This module extends LLM Output Drift findings to multi-step agent trajectories,
answering: When an LLM-based agent executes a finance workflow, what makes it
drift, and what controls make it replayable while staying faithful?

Key Components:

    metrics/
        trajectory_determinism.py  - Trajectory and decision determinism
        faithfulness.py           - Evidence-conditioned faithfulness

    harness/
        stress_test_runner.py     - Drift stress-test harness
        perturbations.py          - Model swap, data shift, quality faults

    tasks/
        (Planned) compliance_triage.py
        (Planned) portfolio_constraint.py
        (Planned) dataops_exception.py

Integration with Existing V2 Infrastructure:
    This module builds on existing harness components:
    - `harness/load_models.py` - Production load testing
    - `metrics/faithfulness.py` - 2x2 determinism matrix (V2)
    - `metrics/semantic_divergence_light.py` - PRSD metrics (V2)
    - `experiments/agentic/compound_drift_analyzer.py` - Multi-step analysis (V2)

Research References:
    - Xiang et al. (2024): Scientific Discovery evaluation
    - OpenAI (2024): Chain-of-Thought Monitorability
    - Ludwig et al. (2024): Econometric Framework for LLMs
    - Yao et al. (2023): ReAct agent architecture

Usage:
    from econometrics.agentic.metrics.trajectory_determinism import (
        AgentTrajectory,
        analyze_trajectory_determinism
    )
    from econometrics.agentic.metrics.faithfulness import (
        compute_faithfulness,
        analyze_frontier
    )
    from econometrics.agentic.harness.stress_test_runner import (
        StressTestHarness,
        create_model_swap_perturbation,
        create_data_shift_perturbation
    )
"""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies
def get_trajectory_metrics():
    """Get trajectory determinism metrics module."""
    from .metrics import trajectory_determinism
    return trajectory_determinism

def get_faithfulness_metrics():
    """Get faithfulness metrics module."""
    from .metrics import faithfulness
    return faithfulness

def get_stress_test_harness():
    """Get stress test harness module."""
    from .harness import stress_test_runner
    return stress_test_runner
