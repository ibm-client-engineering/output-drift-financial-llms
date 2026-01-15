"""
Drift Stress-Test Harness for LLM Agents

A standardized framework to "shake" agents with controlled perturbations
and measure determinism-faithfulness degradation.

Perturbation Categories:
    1. Model/Provider Updates: Version changes, provider swaps
    2. Data Shifts: Stale filings, revised risk factors, schema changes
    3. Data-Quality Faults: Missing fields, inconsistent identifiers
    4. Market Shocks: Rate spikes, volatility jumps, liquidity events

Key Design Principles:
    - Reproducible perturbations via seeded randomness
    - Composable perturbation layers (can combine multiple shocks)
    - Standardized metrics across all perturbation types
    - Trajectory logging for post-hoc analysis

References:
    Xiang, Z., et al. (2024). Evaluating Large Language Models in
    Scientific Discovery. arXiv:2512.15567

    Ludwig, J., Mullainathan, S., & Rambachan, A. (2024).
    Large Language Models: An Applied Econometric Framework.
    arXiv:2412.07031
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Callable, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import copy
import hashlib
from datetime import datetime, timedelta
import logging

# Local imports (relative) - handle both package and direct execution
try:
    from ..metrics.trajectory_determinism import (
        AgentTrajectory, ToolCall,
        analyze_trajectory_determinism, TrajectoryDeterminismMetrics
    )
    from ..metrics.faithfulness import (
        compute_faithfulness, FaithfulnessMetrics,
        AgentDecision, Evidence, PolicyConstraint
    )
except ImportError:
    # Direct execution fallback
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from metrics.trajectory_determinism import (
        AgentTrajectory, ToolCall,
        analyze_trajectory_determinism, TrajectoryDeterminismMetrics
    )
    from metrics.faithfulness import (
        compute_faithfulness, FaithfulnessMetrics,
        AgentDecision, Evidence, PolicyConstraint
    )


logger = logging.getLogger(__name__)


class PerturbationType(Enum):
    """Categories of perturbations for stress testing."""
    BASELINE = "baseline"
    MODEL_SWAP = "model_swap"
    DATA_SHIFT = "data_shift"
    DATA_QUALITY_FAULT = "data_quality_fault"
    MARKET_SHOCK = "market_shock"
    PROMPT_VARIATION = "prompt_variation"
    TEMPORAL_SHIFT = "temporal_shift"


@dataclass
class Perturbation:
    """A specific perturbation to apply during stress testing.

    Attributes:
        perturbation_type: Category of perturbation
        name: Human-readable name
        description: What this perturbation does
        intensity: Severity level (0-1, where 1 is extreme)
        parameters: Perturbation-specific configuration
        apply_fn: Function that applies the perturbation
    """
    perturbation_type: PerturbationType
    name: str
    description: str
    intensity: float = 0.5
    parameters: Dict[str, Any] = field(default_factory=dict)
    apply_fn: Optional[Callable] = None


@dataclass
class StressTestConfig:
    """Configuration for a stress test run.

    Attributes:
        task_name: Name of the task being tested
        n_runs: Number of runs per configuration
        perturbations: List of perturbations to apply
        models: List of models to test
        agent_types: Agent architectures to compare
        random_seed: For reproducibility
        log_trajectories: Whether to save full trajectories
    """
    task_name: str
    n_runs: int = 10
    perturbations: List[Perturbation] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    agent_types: List[str] = field(default_factory=list)
    random_seed: int = 42
    log_trajectories: bool = True


@dataclass
class StressTestResult:
    """Results from a single stress test configuration.

    Attributes:
        config: The configuration used
        perturbation: Perturbation applied (or baseline)
        model: Model used
        agent_type: Agent architecture
        trajectories: All trajectories from runs
        determinism_metrics: Trajectory determinism analysis
        faithfulness_metrics: Per-run faithfulness scores
        degradation: Change from baseline (if applicable)
    """
    config: StressTestConfig
    perturbation: Perturbation
    model: str
    agent_type: str
    trajectories: List[AgentTrajectory] = field(default_factory=list)
    determinism_metrics: Optional[TrajectoryDeterminismMetrics] = None
    faithfulness_metrics: List[FaithfulnessMetrics] = field(default_factory=list)
    degradation: Dict[str, float] = field(default_factory=dict)

    def summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for reporting."""
        return {
            "task": self.config.task_name,
            "perturbation": self.perturbation.name,
            "perturbation_type": self.perturbation.perturbation_type.value,
            "intensity": self.perturbation.intensity,
            "model": self.model,
            "agent_type": self.agent_type,
            "n_runs": len(self.trajectories),
            "action_determinism": self.determinism_metrics.action_determinism if self.determinism_metrics else None,
            "signature_determinism": self.determinism_metrics.signature_determinism if self.determinism_metrics else None,
            "decision_determinism": self.determinism_metrics.decision_determinism if self.determinism_metrics else None,
            "mean_faithfulness": np.mean([m.overall_faithfulness for m in self.faithfulness_metrics]) if self.faithfulness_metrics else None,
            "degradation_determinism": self.degradation.get("determinism", 0),
            "degradation_faithfulness": self.degradation.get("faithfulness", 0)
        }


# ==============================================================================
# Perturbation Implementations
# ==============================================================================

def create_baseline_perturbation() -> Perturbation:
    """Create a no-op baseline perturbation."""
    return Perturbation(
        perturbation_type=PerturbationType.BASELINE,
        name="baseline",
        description="No perturbation - baseline measurement",
        intensity=0.0,
        apply_fn=lambda x: x  # Identity function
    )


def create_model_swap_perturbation(
    original_model: str,
    new_model: str
) -> Perturbation:
    """Create a model/provider swap perturbation.

    This tests robustness to model updates and provider changes.
    From our prior work: Model tier affects determinism dramatically.

    Args:
        original_model: The baseline model
        new_model: The model to swap to

    Returns:
        Perturbation object
    """
    def apply_fn(context: Dict) -> Dict:
        new_context = copy.deepcopy(context)
        new_context['model'] = new_model
        new_context['original_model'] = original_model
        return new_context

    return Perturbation(
        perturbation_type=PerturbationType.MODEL_SWAP,
        name=f"model_swap_{original_model}_to_{new_model}",
        description=f"Swap model from {original_model} to {new_model}",
        intensity=1.0,  # Model swap is a major change
        parameters={"original": original_model, "new": new_model},
        apply_fn=apply_fn
    )


def create_data_shift_perturbation(
    shift_type: str,
    intensity: float = 0.5,
    seed: int = 42
) -> Perturbation:
    """Create a data distribution shift perturbation.

    Types:
    - "stale_data": Use outdated information
    - "schema_change": Modify data schema
    - "value_drift": Shift numerical values
    - "new_entities": Introduce previously unseen entities

    Args:
        shift_type: Type of data shift
        intensity: Severity of shift (0-1)
        seed: Random seed for reproducibility
    """
    rng = np.random.RandomState(seed)

    def apply_fn(context: Dict) -> Dict:
        new_context = copy.deepcopy(context)

        if shift_type == "stale_data":
            # Make data appear older
            if 'data_date' in new_context:
                days_back = int(intensity * 365)  # Up to 1 year old
                new_context['data_date'] = (
                    datetime.now() - timedelta(days=days_back)
                ).isoformat()
            new_context['data_staleness'] = intensity

        elif shift_type == "schema_change":
            # Rename or remove fields
            if 'data' in new_context and isinstance(new_context['data'], dict):
                fields_to_modify = int(len(new_context['data']) * intensity)
                keys = list(new_context['data'].keys())
                for key in rng.choice(keys, min(fields_to_modify, len(keys)), replace=False):
                    if rng.random() < 0.5:
                        # Rename field
                        new_context['data'][f"{key}_v2"] = new_context['data'].pop(key)
                    else:
                        # Remove field
                        del new_context['data'][key]

        elif shift_type == "value_drift":
            # Shift numerical values
            if 'data' in new_context and isinstance(new_context['data'], dict):
                for key, value in new_context['data'].items():
                    if isinstance(value, (int, float)):
                        drift = rng.normal(0, intensity * abs(value))
                        new_context['data'][key] = value + drift

        elif shift_type == "new_entities":
            # Add previously unseen entity references
            new_entities = [f"NEW_ENTITY_{i}" for i in range(int(intensity * 5))]
            new_context['unseen_entities'] = new_entities

        new_context['shift_applied'] = shift_type
        return new_context

    return Perturbation(
        perturbation_type=PerturbationType.DATA_SHIFT,
        name=f"data_shift_{shift_type}",
        description=f"Data shift: {shift_type} with intensity {intensity}",
        intensity=intensity,
        parameters={"shift_type": shift_type, "seed": seed},
        apply_fn=apply_fn
    )


def create_data_quality_fault_perturbation(
    fault_type: str,
    intensity: float = 0.3,
    seed: int = 42
) -> Perturbation:
    """Create a data quality fault perturbation.

    Types:
    - "missing_fields": Remove required fields
    - "type_mismatch": Change data types
    - "inconsistent_ids": Corrupt identifiers
    - "outliers": Inject extreme values
    - "duplicates": Add duplicate records

    Args:
        fault_type: Type of data quality issue
        intensity: Severity (0-1)
        seed: Random seed
    """
    rng = np.random.RandomState(seed)

    def apply_fn(context: Dict) -> Dict:
        new_context = copy.deepcopy(context)

        if 'data' not in new_context:
            new_context['data'] = {}

        if fault_type == "missing_fields":
            # Set some fields to None
            if isinstance(new_context['data'], dict):
                keys = list(new_context['data'].keys())
                n_missing = max(1, int(len(keys) * intensity))
                for key in rng.choice(keys, min(n_missing, len(keys)), replace=False):
                    new_context['data'][key] = None

        elif fault_type == "type_mismatch":
            # Convert types incorrectly
            if isinstance(new_context['data'], dict):
                for key, value in list(new_context['data'].items()):
                    if rng.random() < intensity:
                        if isinstance(value, (int, float)):
                            new_context['data'][key] = str(value)
                        elif isinstance(value, str) and value.isdigit():
                            new_context['data'][key] = int(value)

        elif fault_type == "inconsistent_ids":
            # Corrupt identifier fields
            id_fields = [k for k in new_context['data'].keys()
                        if 'id' in k.lower() or 'identifier' in k.lower()]
            for field in id_fields:
                if rng.random() < intensity:
                    original = new_context['data'][field]
                    if isinstance(original, str):
                        # Add random suffix or prefix
                        new_context['data'][field] = f"{original}_CORRUPTED"

        elif fault_type == "outliers":
            # Inject extreme values
            if isinstance(new_context['data'], dict):
                for key, value in list(new_context['data'].items()):
                    if isinstance(value, (int, float)) and rng.random() < intensity:
                        # Make value 10-100x larger or smaller
                        multiplier = rng.choice([0.01, 0.1, 10, 100])
                        new_context['data'][key] = value * multiplier

        elif fault_type == "duplicates":
            # Add duplicate entries
            new_context['has_duplicates'] = True
            new_context['duplicate_count'] = int(intensity * 5)

        new_context['fault_injected'] = fault_type
        return new_context

    return Perturbation(
        perturbation_type=PerturbationType.DATA_QUALITY_FAULT,
        name=f"dq_fault_{fault_type}",
        description=f"Data quality fault: {fault_type}",
        intensity=intensity,
        parameters={"fault_type": fault_type, "seed": seed},
        apply_fn=apply_fn
    )


def create_market_shock_perturbation(
    shock_type: str,
    intensity: float = 0.5,
    seed: int = 42
) -> Perturbation:
    """Create a market shock perturbation.

    Types:
    - "rate_spike": Interest rate increase
    - "volatility_jump": VIX-style volatility surge
    - "liquidity_crisis": Bid-ask spread widening
    - "correlation_break": Correlation regime change
    - "flash_crash": Sudden price drop

    Args:
        shock_type: Type of market shock
        intensity: Severity (0-1)
        seed: Random seed
    """
    rng = np.random.RandomState(seed)

    def apply_fn(context: Dict) -> Dict:
        new_context = copy.deepcopy(context)

        if 'market_data' not in new_context:
            new_context['market_data'] = {}

        if shock_type == "rate_spike":
            # Simulate interest rate shock
            base_rate = new_context['market_data'].get('interest_rate', 0.05)
            shock_bps = int(intensity * 300)  # Up to 300bps shock
            new_context['market_data']['interest_rate'] = base_rate + shock_bps / 10000
            new_context['market_data']['rate_shock_bps'] = shock_bps

        elif shock_type == "volatility_jump":
            # Simulate VIX spike
            base_vol = new_context['market_data'].get('volatility', 0.20)
            vol_multiplier = 1 + intensity * 3  # Up to 4x volatility
            new_context['market_data']['volatility'] = base_vol * vol_multiplier
            new_context['market_data']['vol_regime'] = 'stressed'

        elif shock_type == "liquidity_crisis":
            # Widen bid-ask spreads
            base_spread = new_context['market_data'].get('spread_bps', 5)
            spread_multiplier = 1 + intensity * 20  # Up to 21x wider
            new_context['market_data']['spread_bps'] = base_spread * spread_multiplier
            new_context['market_data']['liquidity_regime'] = 'crisis'

        elif shock_type == "correlation_break":
            # Correlations go to 1 or -1 (flight to safety / risk-off)
            new_context['market_data']['correlation_regime'] = 'break'
            new_context['market_data']['cross_asset_correlation'] = 0.9 if rng.random() > 0.5 else -0.5

        elif shock_type == "flash_crash":
            # Sudden price drop
            drop_pct = intensity * 15  # Up to 15% drop
            for key in list(new_context['market_data'].keys()):
                if 'price' in key.lower():
                    new_context['market_data'][key] *= (1 - drop_pct / 100)
            new_context['market_data']['flash_crash'] = True
            new_context['market_data']['drop_pct'] = drop_pct

        new_context['shock_applied'] = shock_type
        new_context['shock_intensity'] = intensity
        return new_context

    return Perturbation(
        perturbation_type=PerturbationType.MARKET_SHOCK,
        name=f"market_shock_{shock_type}",
        description=f"Market shock: {shock_type} at intensity {intensity}",
        intensity=intensity,
        parameters={"shock_type": shock_type, "seed": seed},
        apply_fn=apply_fn
    )


# ==============================================================================
# Stress Test Harness
# ==============================================================================

class AgentInterface(ABC):
    """Abstract interface for agents to be stress tested."""

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Tuple[AgentTrajectory, AgentDecision]:
        """Execute the agent on given context.

        Args:
            context: Input context (task description, data, etc.)

        Returns:
            (trajectory, decision) tuple
        """
        pass

    @abstractmethod
    def get_retrieved_evidence(self) -> List[Evidence]:
        """Get evidence retrieved during the last run."""
        pass


class StressTestHarness:
    """Main harness for running stress tests on agents.

    Usage:
        harness = StressTestHarness(config)
        harness.register_agent("schema_first", SchemaFirstAgent())
        harness.register_perturbation(model_swap_perturbation)
        results = harness.run_all()
    """

    def __init__(self, config: StressTestConfig):
        self.config = config
        self.agents: Dict[str, AgentInterface] = {}
        self.perturbations: List[Perturbation] = [create_baseline_perturbation()]
        self.constraints: List[PolicyConstraint] = []
        self.baseline_results: Dict[str, StressTestResult] = {}
        self.all_results: List[StressTestResult] = []

        np.random.seed(config.random_seed)

    def register_agent(self, agent_type: str, agent: AgentInterface):
        """Register an agent for testing."""
        self.agents[agent_type] = agent

    def register_perturbation(self, perturbation: Perturbation):
        """Add a perturbation to the test suite."""
        self.perturbations.append(perturbation)

    def register_constraints(self, constraints: List[PolicyConstraint]):
        """Register policy constraints for faithfulness checking."""
        self.constraints = constraints

    def run_configuration(
        self,
        agent_type: str,
        model: str,
        perturbation: Perturbation,
        base_context: Dict[str, Any]
    ) -> StressTestResult:
        """Run a single configuration n_runs times.

        Args:
            agent_type: Agent architecture to use
            model: Model to use
            perturbation: Perturbation to apply
            base_context: Base input context

        Returns:
            StressTestResult with all metrics
        """
        agent = self.agents.get(agent_type)
        if agent is None:
            raise ValueError(f"Agent type '{agent_type}' not registered")

        trajectories = []
        faithfulness_list = []

        for run_idx in range(self.config.n_runs):
            # Apply perturbation to context
            run_context = copy.deepcopy(base_context)
            run_context['model'] = model
            run_context['run_id'] = f"{agent_type}_{model}_{perturbation.name}_{run_idx}"

            if perturbation.apply_fn:
                run_context = perturbation.apply_fn(run_context)

            # Execute agent
            try:
                trajectory, decision = agent.run(run_context)
                trajectories.append(trajectory)

                # Compute faithfulness
                evidence = agent.get_retrieved_evidence()
                faithfulness = compute_faithfulness(
                    decision, evidence, self.constraints, run_context
                )
                faithfulness_list.append(faithfulness)

            except Exception as e:
                logger.warning(f"Run {run_idx} failed: {e}")
                # Create failure trajectory
                trajectories.append(AgentTrajectory(
                    run_id=run_context['run_id'],
                    input_context=run_context,
                    tool_calls=[],
                    final_decision={"error": str(e)}
                ))

        # Compute determinism metrics
        if len(trajectories) >= 2:
            determinism_metrics = analyze_trajectory_determinism(trajectories)
        else:
            determinism_metrics = None

        result = StressTestResult(
            config=self.config,
            perturbation=perturbation,
            model=model,
            agent_type=agent_type,
            trajectories=trajectories,
            determinism_metrics=determinism_metrics,
            faithfulness_metrics=faithfulness_list
        )

        return result

    def compute_degradation(
        self,
        result: StressTestResult,
        baseline: StressTestResult
    ) -> Dict[str, float]:
        """Compute degradation from baseline.

        Args:
            result: Result under perturbation
            baseline: Baseline result

        Returns:
            Dict with degradation metrics
        """
        degradation = {}

        # Determinism degradation
        if result.determinism_metrics and baseline.determinism_metrics:
            degradation['determinism'] = (
                baseline.determinism_metrics.decision_determinism -
                result.determinism_metrics.decision_determinism
            )

        # Faithfulness degradation
        if result.faithfulness_metrics and baseline.faithfulness_metrics:
            baseline_faith = np.mean([m.overall_faithfulness for m in baseline.faithfulness_metrics])
            result_faith = np.mean([m.overall_faithfulness for m in result.faithfulness_metrics])
            degradation['faithfulness'] = baseline_faith - result_faith

        return degradation

    def run_all(
        self,
        base_context: Dict[str, Any]
    ) -> pd.DataFrame:
        """Run all configurations and return results DataFrame.

        Args:
            base_context: Base input context for the task

        Returns:
            DataFrame with all results
        """
        self.all_results = []
        self.baseline_results = {}

        # First pass: run baselines
        baseline_perturbation = self.perturbations[0]  # Should be baseline
        for agent_type in self.agents.keys():
            for model in self.config.models:
                result = self.run_configuration(
                    agent_type, model, baseline_perturbation, base_context
                )
                self.baseline_results[f"{agent_type}_{model}"] = result
                self.all_results.append(result)

        # Second pass: run perturbations and compute degradation
        for perturbation in self.perturbations[1:]:
            for agent_type in self.agents.keys():
                for model in self.config.models:
                    result = self.run_configuration(
                        agent_type, model, perturbation, base_context
                    )

                    # Compute degradation from baseline
                    baseline_key = f"{agent_type}_{model}"
                    if baseline_key in self.baseline_results:
                        result.degradation = self.compute_degradation(
                            result, self.baseline_results[baseline_key]
                        )

                    self.all_results.append(result)

        # Convert to DataFrame
        rows = [r.summary_dict() for r in self.all_results]
        return pd.DataFrame(rows)

    def generate_report(self) -> str:
        """Generate a human-readable report of stress test results."""
        if not self.all_results:
            return "No results to report. Run stress tests first."

        lines = [
            "=" * 70,
            f"STRESS TEST REPORT: {self.config.task_name}",
            "=" * 70,
            f"Configuration:",
            f"  Runs per config: {self.config.n_runs}",
            f"  Agent types: {list(self.agents.keys())}",
            f"  Models: {self.config.models}",
            f"  Perturbations: {[p.name for p in self.perturbations]}",
            "",
            "-" * 70,
            "BASELINE RESULTS",
            "-" * 70
        ]

        for key, result in self.baseline_results.items():
            det = result.determinism_metrics
            faith = result.faithfulness_metrics
            lines.append(f"\n{key}:")
            if det:
                lines.append(f"  Action Determinism:    {det.action_determinism:.1%}")
                lines.append(f"  Signature Determinism: {det.signature_determinism:.1%}")
                lines.append(f"  Decision Determinism:  {det.decision_determinism:.1%}")
            if faith:
                mean_faith = np.mean([m.overall_faithfulness for m in faith])
                lines.append(f"  Mean Faithfulness:     {mean_faith:.1%}")

        lines.extend([
            "",
            "-" * 70,
            "DEGRADATION UNDER PERTURBATIONS",
            "-" * 70
        ])

        # Group by perturbation
        by_perturbation = {}
        for result in self.all_results:
            if result.perturbation.name == "baseline":
                continue
            key = result.perturbation.name
            if key not in by_perturbation:
                by_perturbation[key] = []
            by_perturbation[key].append(result)

        for pert_name, results in by_perturbation.items():
            lines.append(f"\nPerturbation: {pert_name}")
            for result in results:
                if result.degradation:
                    lines.append(f"  {result.agent_type} / {result.model}:")
                    det_deg = result.degradation.get('determinism', 0)
                    faith_deg = result.degradation.get('faithfulness', 0)
                    lines.append(f"    Determinism degradation:  {det_deg:+.1%}")
                    lines.append(f"    Faithfulness degradation: {faith_deg:+.1%}")

        lines.extend([
            "",
            "=" * 70,
            "RECOMMENDATIONS",
            "=" * 70
        ])

        # Find most robust configurations
        robust_configs = []
        for result in self.all_results:
            if result.perturbation.name != "baseline" and result.degradation:
                total_deg = abs(result.degradation.get('determinism', 0)) + \
                           abs(result.degradation.get('faithfulness', 0))
                robust_configs.append((result.agent_type, result.model, total_deg))

        robust_configs.sort(key=lambda x: x[2])

        lines.append("\nMost robust configurations (lowest total degradation):")
        for agent_type, model, deg in robust_configs[:3]:
            lines.append(f"  {agent_type} / {model}: {deg:.1%} total degradation")

        return "\n".join(lines)


# ==============================================================================
# Example: Compliance Triage Stress Test
# ==============================================================================

class MockComplianceAgent(AgentInterface):
    """Mock compliance agent for demonstration."""

    def __init__(self, determinism_level: float = 0.9):
        self.determinism_level = determinism_level
        self.last_evidence = []
        self.rng = np.random.RandomState(42)

    def run(self, context: Dict[str, Any]) -> Tuple[AgentTrajectory, AgentDecision]:
        """Run mock compliance triage."""
        run_id = context.get('run_id', 'mock_run')

        # Simulate tool calls
        tool_calls = [
            ToolCall("retrieve_rules", {"alert_type": context.get('alert_type', 'unknown')}),
            ToolCall("check_policy", {"entity": context.get('entity', 'unknown')}),
        ]

        # Add variation based on determinism level
        if self.rng.random() > self.determinism_level:
            tool_calls.append(ToolCall("search_precedents", {"entity": context.get('entity')}))

        tool_calls.append(ToolCall("classify_alert", {"severity": "high"}))
        tool_calls.append(ToolCall("generate_trail", {"classification": "escalate"}))

        # Simulate evidence
        self.last_evidence = [
            Evidence("rule_001", "rules_db", "SEC Rule 10b-5: Prohibition on insider trading"),
            Evidence("policy_001", "policy_db", "Entity monitoring policy requires escalation for unusual activity")
        ]

        trajectory = AgentTrajectory(
            run_id=run_id,
            input_context=context,
            tool_calls=tool_calls,
            final_decision={"action": "escalate", "priority": 1}
        )

        decision = AgentDecision(
            decision_id=run_id,
            action={"action": "escalate", "priority": 1},
            justification="Based on rule_001 and policy_001, this alert requires escalation.",
            cited_evidence=["rule_001", "policy_001"]
        )

        return trajectory, decision

    def get_retrieved_evidence(self) -> List[Evidence]:
        return self.last_evidence


def example_stress_test():
    """Demonstrate the stress test harness."""
    print("="*70)
    print("STRESS TEST HARNESS DEMONSTRATION")
    print("="*70)
    print()

    # Create configuration
    config = StressTestConfig(
        task_name="Compliance Triage Agent",
        n_runs=5,
        models=["gpt-4o", "claude-opus"],
        agent_types=["unconstrained", "schema_first"],
        random_seed=42
    )

    # Create harness
    harness = StressTestHarness(config)

    # Register agents
    harness.register_agent("unconstrained", MockComplianceAgent(determinism_level=0.6))
    harness.register_agent("schema_first", MockComplianceAgent(determinism_level=0.95))

    # Register perturbations
    harness.register_perturbation(
        create_model_swap_perturbation("gpt-4o", "claude-opus")
    )
    harness.register_perturbation(
        create_data_shift_perturbation("stale_data", intensity=0.5)
    )
    harness.register_perturbation(
        create_data_quality_fault_perturbation("missing_fields", intensity=0.3)
    )
    harness.register_perturbation(
        create_market_shock_perturbation("volatility_jump", intensity=0.7)
    )

    # Define base context
    base_context = {
        "alert_id": "SEC-2024-001",
        "alert_type": "insider_trading_suspicion",
        "entity": "ACME Corp",
        "description": "Unusual options activity before earnings",
        "data_date": datetime.now().isoformat(),
        "data": {
            "volume": 150000,
            "avg_volume": 50000,
            "price_change_pct": 5.2,
            "entity_id": "ACME_CORP_001"
        }
    }

    # Run stress tests
    print("Running stress tests...")
    results_df = harness.run_all(base_context)

    # Print results
    print("\nResults DataFrame:")
    print(results_df.to_string())

    # Generate report
    print("\n" + harness.generate_report())


if __name__ == "__main__":
    example_stress_test()
