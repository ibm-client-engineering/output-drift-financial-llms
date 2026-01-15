"""
Drift Stress-Test Harness for LLM Agents

A standardized framework to "shake" agents with controlled perturbations
and measure determinism-faithfulness degradation.

Perturbation Categories:
    1. Model/Provider Updates - Version changes, provider swaps
    2. Data Shifts - Stale filings, revised risk factors, schema changes
    3. Data-Quality Faults - Missing fields, inconsistent identifiers
    4. Market Shocks - Rate spikes, volatility jumps, liquidity events

Integration with V2 Infrastructure:
    This harness builds on existing V2 components:
    - `harness/load_models.py` (V2) - Load testing orchestrator
    - `providers/watsonx.py` (V2) - watsonx.ai provider
    - `harness/cross_provider_validation.py` (V2) - Cross-provider testing

Design Principles:
    - Reproducible perturbations via seeded randomness
    - Composable perturbation layers (can combine multiple shocks)
    - Standardized metrics across all perturbation types
    - Trajectory logging for post-hoc analysis
"""

from .stress_test_runner import (
    PerturbationType,
    Perturbation,
    StressTestConfig,
    StressTestResult,
    StressTestHarness,
    AgentInterface,
    # Perturbation factories
    create_baseline_perturbation,
    create_model_swap_perturbation,
    create_data_shift_perturbation,
    create_data_quality_fault_perturbation,
    create_market_shock_perturbation,
)

__all__ = [
    'PerturbationType',
    'Perturbation',
    'StressTestConfig',
    'StressTestResult',
    'StressTestHarness',
    'AgentInterface',
    'create_baseline_perturbation',
    'create_model_swap_perturbation',
    'create_data_shift_perturbation',
    'create_data_quality_fault_perturbation',
    'create_market_shock_perturbation',
]
