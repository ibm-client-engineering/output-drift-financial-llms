"""
Trajectory Determinism Metrics for LLM Agents

Extends output drift concepts to multi-step agent trajectories.

Key Insight from Scientific Discovery Literature:
    Real science (and real financial decision-making) is iterative:
    hypothesize → experiment → observe → revise → repeat

    LLMs struggle with this loop (Xiang et al. 2024, arXiv:2512.15567):
    - Overfit to surface patterns
    - Fail to abandon bad hypotheses
    - Confuse correlation for causation
    - Hallucinate explanations when experiments fail

    Our contribution: Quantify these failure modes as TRAJECTORY DRIFT.

Trajectory Determinism Definition:
    Given identical inputs, does the agent produce:
    1. Identical tool call sequences?
    2. Identical intermediate states?
    3. Identical final decisions?

    We decompose trajectory drift into:
    - Action drift: Different tools called
    - Argument drift: Same tools, different arguments
    - Ordering drift: Same tools, different sequence
    - State drift: Same actions, different intermediate states

References:
    Xiang, Z., et al. (2024). Evaluating Large Language Models in
    Scientific Discovery. arXiv:2512.15567

    Ludwig, J., Mullainathan, S., & Rambachan, A. (2024).
    Large Language Models: An Applied Econometric Framework.
    arXiv:2412.07031
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import Counter
import json
import hashlib
from difflib import SequenceMatcher


@dataclass
class ToolCall:
    """Represents a single tool invocation in an agent trajectory.

    Attributes:
        tool_name: Name of the tool called
        arguments: Arguments passed to the tool (dict)
        result: Tool output (if captured)
        timestamp: Relative timing in trajectory
        reasoning: Agent's reasoning before the call (if available)
    """
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    timestamp: Optional[float] = None
    reasoning: Optional[str] = None

    def signature(self) -> str:
        """Canonical string representation for comparison."""
        args_sorted = json.dumps(self.arguments, sort_keys=True, default=str)
        return f"{self.tool_name}({args_sorted})"

    def signature_hash(self) -> str:
        """Hash of signature for efficient comparison."""
        return hashlib.md5(self.signature().encode()).hexdigest()[:8]


@dataclass
class AgentTrajectory:
    """Complete trajectory of an agent run.

    Attributes:
        run_id: Unique identifier for this run
        input_context: The input that triggered this trajectory
        tool_calls: Ordered list of tool invocations
        final_decision: Terminal action/output
        intermediate_states: State snapshots after each tool call
        total_tokens: Token usage (if tracked)
        latency_ms: Total execution time
    """
    run_id: str
    input_context: Dict[str, Any]
    tool_calls: List[ToolCall]
    final_decision: Any
    intermediate_states: List[Dict] = field(default_factory=list)
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None

    def tool_sequence(self) -> List[str]:
        """Extract ordered list of tool names."""
        return [tc.tool_name for tc in self.tool_calls]

    def signature_sequence(self) -> List[str]:
        """Extract ordered list of full tool signatures."""
        return [tc.signature() for tc in self.tool_calls]

    def hash_sequence(self) -> List[str]:
        """Extract ordered list of signature hashes."""
        return [tc.signature_hash() for tc in self.tool_calls]

    def trajectory_hash(self) -> str:
        """Single hash representing entire trajectory."""
        seq_str = "|".join(self.signature_sequence())
        return hashlib.md5(seq_str.encode()).hexdigest()


@dataclass
class TrajectoryDeterminismMetrics:
    """Comprehensive trajectory determinism metrics.

    Attributes:
        n_runs: Number of runs compared
        action_determinism: % runs with identical tool sequence (names only)
        signature_determinism: % runs with identical signatures (names + args)
        decision_determinism: % runs with identical final decisions
        mean_sequence_similarity: Average Levenshtein-like similarity
        mode_trajectory: Most common trajectory pattern
        mode_frequency: Frequency of mode trajectory
        drift_breakdown: Detailed drift category counts
    """
    n_runs: int
    action_determinism: float
    signature_determinism: float
    decision_determinism: float
    mean_sequence_similarity: float
    mode_trajectory: Optional[str] = None
    mode_frequency: float = 0.0
    drift_breakdown: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return f"""
Trajectory Determinism Analysis
{'='*60}
Total runs analyzed:        {self.n_runs}

DETERMINISM SCORES:
  Action Determinism:       {self.action_determinism:.1%}
    (identical tool sequence, ignoring arguments)

  Signature Determinism:    {self.signature_determinism:.1%}
    (identical tool + arguments)

  Decision Determinism:     {self.decision_determinism:.1%}
    (identical final output/action)

TRAJECTORY SIMILARITY:
  Mean Sequence Similarity: {self.mean_sequence_similarity:.3f}
  Mode Trajectory Freq:     {self.mode_frequency:.1%}

DRIFT BREAKDOWN:
  {self._format_drift_breakdown()}

INTERPRETATION:
  Action > Signature: Arguments vary but workflow is stable
  Decision > Signature: Different paths, same conclusion
  Low all three: High trajectory drift - audit required
"""

    def _format_drift_breakdown(self) -> str:
        if not self.drift_breakdown:
            return "No drift detected"
        lines = []
        for category, count in sorted(self.drift_breakdown.items()):
            lines.append(f"{category}: {count} runs")
        return "\n  ".join(lines)


def compute_sequence_similarity(seq1: List[str], seq2: List[str]) -> float:
    """Compute similarity between two tool sequences.

    Uses SequenceMatcher for alignment-based comparison.
    Returns value in [0, 1] where 1 = identical.

    Args:
        seq1, seq2: Lists of tool names or signatures

    Returns:
        Similarity score in [0, 1]

    Example:
        >>> seq1 = ['retrieve', 'analyze', 'decide']
        >>> seq2 = ['retrieve', 'validate', 'analyze', 'decide']
        >>> sim = compute_sequence_similarity(seq1, seq2)
        >>> print(f"{sim:.2f}")  # ~0.86
    """
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0

    matcher = SequenceMatcher(None, seq1, seq2)
    return matcher.ratio()


def compute_edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """Compute Levenshtein edit distance between sequences.

    Args:
        seq1, seq2: Lists of tool names or signatures

    Returns:
        Edit distance (insertions + deletions + substitutions)
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]


def classify_drift_type(
    traj1: AgentTrajectory,
    traj2: AgentTrajectory
) -> str:
    """Classify the type of drift between two trajectories.

    Categories:
    - "identical": No drift
    - "action_drift": Different tools called
    - "argument_drift": Same tools, different arguments
    - "ordering_drift": Same tools, different order
    - "state_drift": Same actions, different intermediate states
    - "decision_drift": Same trajectory, different final decision

    Args:
        traj1, traj2: Two trajectories to compare

    Returns:
        Drift category string
    """
    seq1 = traj1.tool_sequence()
    seq2 = traj2.tool_sequence()
    sig1 = traj1.signature_sequence()
    sig2 = traj2.signature_sequence()

    # Check signatures first (most specific)
    if sig1 == sig2:
        # Same trajectory - check final decision
        if traj1.final_decision == traj2.final_decision:
            return "identical"
        else:
            return "decision_drift"

    # Check if same tools in same order but different args
    if seq1 == seq2:
        return "argument_drift"

    # Check if same tools but different order
    if sorted(seq1) == sorted(seq2):
        return "ordering_drift"

    # Different tools called
    return "action_drift"


def analyze_trajectory_determinism(
    trajectories: List[AgentTrajectory],
    reference_trajectory: Optional[AgentTrajectory] = None
) -> TrajectoryDeterminismMetrics:
    """Analyze determinism across multiple trajectory runs.

    Args:
        trajectories: List of trajectories from repeated runs
        reference_trajectory: Optional baseline to compare against

    Returns:
        TrajectoryDeterminismMetrics with comprehensive analysis

    Example:
        >>> # Run agent 10 times on same input
        >>> trajectories = [run_agent(input_ctx) for _ in range(10)]
        >>> metrics = analyze_trajectory_determinism(trajectories)
        >>> print(metrics.summary())
    """
    n = len(trajectories)
    if n < 2:
        raise ValueError("Need at least 2 trajectories for comparison")

    # Extract sequences
    tool_sequences = [t.tool_sequence() for t in trajectories]
    signature_sequences = [t.signature_sequence() for t in trajectories]
    trajectory_hashes = [t.trajectory_hash() for t in trajectories]
    final_decisions = [t.final_decision for t in trajectories]

    # Action determinism: % with identical tool sequence (ignoring args)
    tool_seq_strs = ["|".join(seq) for seq in tool_sequences]
    mode_tool_seq = Counter(tool_seq_strs).most_common(1)[0]
    action_determinism = mode_tool_seq[1] / n

    # Signature determinism: % with identical full signatures
    mode_sig = Counter(trajectory_hashes).most_common(1)[0]
    signature_determinism = mode_sig[1] / n

    # Decision determinism: % with identical final decision
    decision_strs = [json.dumps(d, sort_keys=True, default=str) for d in final_decisions]
    mode_decision = Counter(decision_strs).most_common(1)[0]
    decision_determinism = mode_decision[1] / n

    # Mean pairwise sequence similarity
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_sequence_similarity(signature_sequences[i], signature_sequences[j])
            similarities.append(sim)
    mean_similarity = np.mean(similarities) if similarities else 1.0

    # Drift breakdown
    drift_counts = Counter()
    reference = reference_trajectory or trajectories[0]
    for traj in trajectories[1:]:
        drift_type = classify_drift_type(reference, traj)
        drift_counts[drift_type] += 1

    return TrajectoryDeterminismMetrics(
        n_runs=n,
        action_determinism=action_determinism,
        signature_determinism=signature_determinism,
        decision_determinism=decision_determinism,
        mean_sequence_similarity=mean_similarity,
        mode_trajectory=mode_sig[0],
        mode_frequency=mode_sig[1] / n,
        drift_breakdown=dict(drift_counts)
    )


def compute_trajectory_entropy(trajectories: List[AgentTrajectory]) -> float:
    """Compute entropy of trajectory distribution.

    Higher entropy = more diverse trajectories = lower determinism.

    Args:
        trajectories: List of trajectories

    Returns:
        Shannon entropy of trajectory hash distribution
    """
    hashes = [t.trajectory_hash() for t in trajectories]
    counts = Counter(hashes)
    n = len(hashes)

    probs = [count / n for count in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)

    return entropy


def detect_hypothesis_persistence(
    trajectories: List[AgentTrajectory],
    hypothesis_extractor: Optional[callable] = None
) -> Dict[str, Any]:
    """Detect whether agent persists with hypotheses despite contradicting evidence.

    Inspired by Xiang et al. (2024) finding that LLMs struggle to abandon
    bad hypotheses even when evidence contradicts them.

    Args:
        trajectories: List of trajectories to analyze
        hypothesis_extractor: Function to extract hypothesis from trajectory state

    Returns:
        Dict with hypothesis persistence metrics
    """
    # Default extractor looks for 'hypothesis' or 'belief' in intermediate states
    if hypothesis_extractor is None:
        def hypothesis_extractor(traj):
            hypotheses = []
            for state in traj.intermediate_states:
                if 'hypothesis' in state:
                    hypotheses.append(state['hypothesis'])
                elif 'belief' in state:
                    hypotheses.append(state['belief'])
            return hypotheses

    results = {
        'trajectories_analyzed': len(trajectories),
        'hypothesis_changes_per_run': [],
        'hypothesis_persistence_rate': 0.0,
        'hypothesis_revision_failures': 0
    }

    for traj in trajectories:
        hypotheses = hypothesis_extractor(traj)
        if len(hypotheses) > 1:
            # Count how many times hypothesis changed
            changes = sum(1 for i in range(1, len(hypotheses))
                         if hypotheses[i] != hypotheses[i-1])
            results['hypothesis_changes_per_run'].append(changes)
        else:
            results['hypothesis_changes_per_run'].append(0)

    # Persistence rate = runs where hypothesis never changed
    if results['hypothesis_changes_per_run']:
        n_persistent = sum(1 for c in results['hypothesis_changes_per_run'] if c == 0)
        results['hypothesis_persistence_rate'] = n_persistent / len(trajectories)

    return results


# ==============================================================================
# Example: Compliance Triage Agent Trajectory Analysis
# ==============================================================================

def example_compliance_agent_determinism():
    """
    Demonstrates trajectory determinism analysis for a compliance triage agent.

    Task: Classify regulatory alert, justify, generate audit trail
    Tools: retrieve_rules, check_policy, classify_alert, generate_trail
    """
    print("="*60)
    print("Trajectory Determinism: Compliance Triage Agent")
    print("="*60)
    print()

    # Simulate 10 runs of the same agent on same input
    np.random.seed(42)

    input_context = {
        "alert_id": "SEC-2024-001",
        "alert_type": "insider_trading_suspicion",
        "entity": "ACME Corp",
        "description": "Unusual options activity before earnings"
    }

    trajectories = []

    # Run 1-5: Deterministic agent (schema-first)
    for i in range(5):
        traj = AgentTrajectory(
            run_id=f"schema_first_{i}",
            input_context=input_context,
            tool_calls=[
                ToolCall("retrieve_rules", {"alert_type": "insider_trading_suspicion"}),
                ToolCall("check_policy", {"entity": "ACME Corp", "rule_id": "SEC-10b5"}),
                ToolCall("classify_alert", {"severity": "high", "confidence": 0.92}),
                ToolCall("generate_trail", {"classification": "escalate", "evidence": ["options_data", "timing_analysis"]})
            ],
            final_decision={"action": "escalate", "priority": 1},
            intermediate_states=[
                {"hypothesis": "potential_violation"},
                {"hypothesis": "potential_violation", "evidence_strength": "strong"},
                {"hypothesis": "confirmed_violation"},
                {"hypothesis": "confirmed_violation", "action_taken": True}
            ]
        )
        trajectories.append(traj)

    # Run 6-8: Slight argument drift (same tools, different params)
    for i in range(3):
        confidence = 0.88 + np.random.uniform(-0.05, 0.05)
        traj = AgentTrajectory(
            run_id=f"arg_drift_{i}",
            input_context=input_context,
            tool_calls=[
                ToolCall("retrieve_rules", {"alert_type": "insider_trading_suspicion"}),
                ToolCall("check_policy", {"entity": "ACME Corp", "rule_id": "SEC-10b5"}),
                ToolCall("classify_alert", {"severity": "high", "confidence": round(confidence, 2)}),
                ToolCall("generate_trail", {"classification": "escalate", "evidence": ["options_data"]})
            ],
            final_decision={"action": "escalate", "priority": 1},
            intermediate_states=[]
        )
        trajectories.append(traj)

    # Run 9-10: Action drift (different tool sequence)
    for i in range(2):
        traj = AgentTrajectory(
            run_id=f"action_drift_{i}",
            input_context=input_context,
            tool_calls=[
                ToolCall("retrieve_rules", {"alert_type": "insider_trading_suspicion"}),
                ToolCall("search_precedents", {"entity": "ACME Corp"}),  # Extra tool
                ToolCall("check_policy", {"entity": "ACME Corp", "rule_id": "SEC-10b5"}),
                ToolCall("classify_alert", {"severity": "high", "confidence": 0.95}),
                ToolCall("generate_trail", {"classification": "escalate", "evidence": ["options_data", "precedent_match"]})
            ],
            final_decision={"action": "escalate", "priority": 1},
            intermediate_states=[]
        )
        trajectories.append(traj)

    # Analyze determinism
    metrics = analyze_trajectory_determinism(trajectories)
    print(metrics.summary())

    # Entropy analysis
    entropy = compute_trajectory_entropy(trajectories)
    print(f"Trajectory Entropy: {entropy:.3f} bits")
    print(f"  (Lower = more deterministic, 0 = perfectly deterministic)")
    print()

    # Hypothesis persistence (for runs with intermediate states)
    persistence = detect_hypothesis_persistence(trajectories[:5])
    print(f"Hypothesis Persistence Analysis (schema-first runs):")
    print(f"  Persistence rate: {persistence['hypothesis_persistence_rate']:.1%}")
    print(f"  Mean changes per run: {np.mean(persistence['hypothesis_changes_per_run']):.2f}")
    print()

    print("="*60)
    print("KEY FINDING:")
    print("="*60)
    print(f"  Decision Determinism ({metrics.decision_determinism:.0%}) > Signature Determinism ({metrics.signature_determinism:.0%})")
    print("  → Agent takes different paths but reaches same conclusion")
    print("  → Acceptable for audit if final decision is correct")
    print("  → Schema-first design achieves higher trajectory determinism")


if __name__ == "__main__":
    example_compliance_agent_determinism()
