"""
Agent Replayability Metrics

Three core metrics for measuring agent replayability:

1. Trajectory Determinism
   - Identical tool-call sequence + arguments + intermediate states
   - Measured via edit distance on action traces

2. Decision Determinism
   - Identical final action/decision even if internal reasoning varies
   - Measured via exact match on terminal action

3. Evidence-Conditioned Faithfulness
   - Decision is supported by retrieved/validated evidence + policy constraints
   - Measured via LLM-as-Judge + rule validation

Integration with V2 Metrics:
    This module complements existing V2 metrics:
    - `metrics/faithfulness.py` (V2) - 2x2 determinism matrix
    - `metrics/semantic_divergence_light.py` (V2) - PRSD framework
    - `metrics/stats.py` (V2) - Statistical rigor (FDR, power analysis)

Key Difference from V2:
    V2 measures OUTPUT drift (single LLM call)
    This module measures TRAJECTORY drift (multi-step agent)
"""

from .trajectory_determinism import (
    ToolCall,
    AgentTrajectory,
    TrajectoryDeterminismMetrics,
    analyze_trajectory_determinism,
    compute_trajectory_entropy,
    classify_drift_type
)

from .faithfulness import (
    Evidence,
    PolicyConstraint,
    AgentDecision,
    FaithfulnessMetrics,
    DeterminismFaithfulnessPoint,
    compute_faithfulness,
    compute_frontier,
    analyze_frontier
)

__all__ = [
    # Trajectory
    'ToolCall',
    'AgentTrajectory',
    'TrajectoryDeterminismMetrics',
    'analyze_trajectory_determinism',
    'compute_trajectory_entropy',
    'classify_drift_type',
    # Faithfulness
    'Evidence',
    'PolicyConstraint',
    'AgentDecision',
    'FaithfulnessMetrics',
    'DeterminismFaithfulnessPoint',
    'compute_faithfulness',
    'compute_frontier',
    'analyze_frontier',
]
