"""
Evidence-Conditioned Faithfulness Metrics for LLM Agents

Measures whether agent decisions are grounded in retrieved evidence
and satisfy policy constraints.

Key Insight from Scientific Discovery Literature (Xiang et al. 2024):
    LLMs "optimize for plausibility, not truth" - they generate
    convincing explanations even when experiments fail or evidence
    contradicts their claims.

    For financial agents, this manifests as:
    - Hallucinated justifications for trading decisions
    - Fabricated citations to non-existent regulations
    - Confident explanations that ignore retrieved evidence
    - Policy violations masked by fluent reasoning

Halperin (Dec 2025) Critical Insight - Information-Theoretic Faithfulness:
    From Fidelity Investments research on financial disclosures (arXiv:2512.05156):

    "An answer can look correct, sound well, and still be structurally
     misaligned with the question and context."

    The metric penalized answers that quietly made up entities even when
    LLM-based judges rated them as "good and complete."

    Key concepts:
    - Faithfulness = minimal KL divergence between question→topic and answer→topic
    - Semantic Entropy Production = irreversibility/noise in generation
    - Hallucinations should be MANAGED, not eliminated (psychiatric terminology)

    Code: https://github.com/ighalp/semantic-faithfulness-sdm

Our Contribution:
    Evidence-conditioned faithfulness decomposes into:
    1. Evidence Grounding: Does the decision cite real, retrieved evidence?
    2. Evidence Alignment: Does the decision follow from the evidence?
    3. Constraint Satisfaction: Does the decision respect policy constraints?
    4. Justification Validity: Is the stated reasoning traceable to evidence?

Relationship to Output Drift:
    From our prior work:
    - Faithfulness and determinism are ORTHOGONAL dimensions
    - "Varying-but-correct" (low determinism, high faithfulness) is ACCEPTABLE
    - "Consistent-but-wrong" (high determinism, low faithfulness) is DANGEROUS

    This module operationalizes faithfulness for agent trajectories.

References:
    Halperin, I. (2025). Semantic Divergence Metrics to Manage Hallucinations
    in Large Language Models. arXiv:2512.05156 (December 2025)
    Code: https://github.com/ighalp/semantic-faithfulness-sdm

    Xiang, Z., et al. (2024). Evaluating Large Language Models in
    Scientific Discovery. arXiv:2512.15567

    Ludwig, J., Mullainathan, S., & Rambachan, A. (2024).
    Large Language Models: An Applied Econometric Framework.
    arXiv:2412.07031
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Evidence:
    """A piece of evidence retrieved or generated during agent execution.

    Attributes:
        evidence_id: Unique identifier
        source: Where this evidence came from (tool name, document, etc.)
        content: The actual evidence content
        timestamp: When this was retrieved
        metadata: Additional context (document ID, page, confidence, etc.)
    """
    evidence_id: str
    source: str
    content: str
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Create a unique fingerprint for this evidence."""
        return f"{self.source}:{self.evidence_id}"


@dataclass
class PolicyConstraint:
    """A policy constraint that decisions must satisfy.

    Attributes:
        constraint_id: Unique identifier
        description: Human-readable description
        validator: Function that checks if a decision satisfies this constraint
        severity: "hard" (must satisfy) or "soft" (should satisfy)
        category: Type of constraint (regulatory, risk, operational, etc.)
    """
    constraint_id: str
    description: str
    validator: Callable[[Any, Dict], bool]
    severity: str = "hard"  # "hard" or "soft"
    category: str = "general"


@dataclass
class FaithfulnessMetrics:
    """Comprehensive faithfulness metrics for an agent decision.

    Attributes:
        evidence_grounding_score: % of cited evidence that was actually retrieved
        evidence_alignment_score: Semantic similarity between decision and evidence
        constraint_satisfaction_rate: % of constraints satisfied
        justification_validity_score: How well reasoning traces to evidence
        hallucination_rate: % of claims not grounded in evidence
        overall_faithfulness: Weighted combination of above scores
    """
    evidence_grounding_score: float
    evidence_alignment_score: float
    constraint_satisfaction_rate: float
    justification_validity_score: float
    hallucination_rate: float
    overall_faithfulness: float

    # Detailed breakdowns
    grounding_details: Dict[str, bool] = field(default_factory=dict)
    constraint_details: Dict[str, bool] = field(default_factory=dict)
    hallucinated_claims: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"""
Faithfulness Analysis
{'='*60}
CORE METRICS:
  Evidence Grounding:       {self.evidence_grounding_score:.1%}
    (% of cited evidence actually retrieved)

  Evidence Alignment:       {self.evidence_alignment_score:.1%}
    (semantic match between decision and evidence)

  Constraint Satisfaction:  {self.constraint_satisfaction_rate:.1%}
    (% of policy constraints satisfied)

  Justification Validity:   {self.justification_validity_score:.1%}
    (reasoning traces to evidence)

  Hallucination Rate:       {self.hallucination_rate:.1%}
    (claims not grounded in evidence)

OVERALL FAITHFULNESS:       {self.overall_faithfulness:.1%}

INTERPRETATION:
  ≥90%: Excellent - auditable, trustworthy decision
  70-90%: Good - minor gaps, acceptable with review
  50-70%: Moderate - requires human verification
  <50%: Poor - decision cannot be trusted

CONSTRAINT VIOLATIONS:
  {self._format_violations()}

HALLUCINATED CLAIMS:
  {self._format_hallucinations()}
"""

    def _format_violations(self) -> str:
        violations = [cid for cid, satisfied in self.constraint_details.items() if not satisfied]
        if not violations:
            return "None"
        return "\n  ".join(violations)

    def _format_hallucinations(self) -> str:
        if not self.hallucinated_claims:
            return "None detected"
        return "\n  ".join(self.hallucinated_claims[:5])  # Show first 5


@dataclass
class AgentDecision:
    """An agent's final decision with supporting information.

    Attributes:
        decision_id: Unique identifier
        action: The actual decision/action taken
        justification: Agent's stated reasoning
        cited_evidence: Evidence IDs the agent claims to use
        confidence: Agent's stated confidence (if provided)
        metadata: Additional context
    """
    decision_id: str
    action: Any
    justification: str
    cited_evidence: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_evidence_grounding(
    decision: AgentDecision,
    retrieved_evidence: List[Evidence]
) -> Tuple[float, Dict[str, bool]]:
    """Check if cited evidence was actually retrieved.

    A decision is grounded if all cited evidence IDs correspond to
    evidence that was actually retrieved during the trajectory.

    Args:
        decision: The agent's decision with cited evidence
        retrieved_evidence: Evidence actually retrieved

    Returns:
        (grounding_score, details_dict)

    Example:
        >>> decision = AgentDecision(
        ...     decision_id="d1",
        ...     action="escalate",
        ...     justification="Based on SEC Rule 10b-5...",
        ...     cited_evidence=["SEC-10b5", "precedent-2023-001", "fake-citation"]
        ... )
        >>> evidence = [Evidence("SEC-10b5", "rules_db", "..."),
        ...             Evidence("precedent-2023-001", "case_db", "...")]
        >>> score, details = compute_evidence_grounding(decision, evidence)
        >>> print(score)  # 0.67 (2/3 citations are real)
    """
    if not decision.cited_evidence:
        # No citations = cannot verify grounding
        return 0.0, {}

    retrieved_ids = {e.evidence_id for e in retrieved_evidence}
    details = {}

    grounded_count = 0
    for cited_id in decision.cited_evidence:
        is_grounded = cited_id in retrieved_ids
        details[cited_id] = is_grounded
        if is_grounded:
            grounded_count += 1

    score = grounded_count / len(decision.cited_evidence)
    return score, details


def compute_evidence_alignment(
    decision: AgentDecision,
    retrieved_evidence: List[Evidence],
    vectorizer: Optional[TfidfVectorizer] = None
) -> float:
    """Measure semantic similarity between decision and cited evidence.

    High alignment = decision content matches evidence content.
    Low alignment = decision says things not in evidence (potential hallucination).

    Args:
        decision: The agent's decision
        retrieved_evidence: Retrieved evidence list
        vectorizer: Optional TF-IDF vectorizer

    Returns:
        Alignment score in [0, 1]
    """
    if not retrieved_evidence:
        return 0.0

    # Combine decision text
    decision_text = f"{decision.action} {decision.justification}"

    # Combine evidence text
    evidence_texts = [e.content for e in retrieved_evidence]
    combined_evidence = " ".join(evidence_texts)

    if not combined_evidence.strip():
        return 0.0

    # Compute TF-IDF similarity
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))

    try:
        corpus = [decision_text, combined_evidence]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0]
        return float(similarity)
    except Exception:
        return 0.0


def check_constraint_satisfaction(
    decision: AgentDecision,
    constraints: List[PolicyConstraint],
    context: Dict[str, Any]
) -> Tuple[float, Dict[str, bool]]:
    """Check if decision satisfies all policy constraints.

    Args:
        decision: The agent's decision
        constraints: List of policy constraints to check
        context: Additional context for constraint evaluation

    Returns:
        (satisfaction_rate, constraint_details)

    Example:
        >>> def max_position_check(action, ctx):
        ...     return action.get('position_size', 0) <= ctx.get('max_position', 1000)
        >>> constraint = PolicyConstraint("max_position", "Position size limit",
        ...                               max_position_check, "hard", "risk")
        >>> decision = AgentDecision("d1", {"position_size": 500}, "Buy signal")
        >>> rate, details = check_constraint_satisfaction(decision, [constraint], {"max_position": 1000})
        >>> print(rate)  # 1.0 (satisfied)
    """
    if not constraints:
        return 1.0, {}

    details = {}
    satisfied_count = 0
    hard_violations = 0

    for constraint in constraints:
        try:
            is_satisfied = constraint.validator(decision.action, context)
            details[constraint.constraint_id] = is_satisfied

            if is_satisfied:
                satisfied_count += 1
            elif constraint.severity == "hard":
                hard_violations += 1
        except Exception as e:
            # Constraint check failed - treat as violation
            details[constraint.constraint_id] = False

    # If any hard constraint is violated, overall satisfaction is 0
    if hard_violations > 0:
        return 0.0, details

    satisfaction_rate = satisfied_count / len(constraints)
    return satisfaction_rate, details


def detect_hallucinations(
    decision: AgentDecision,
    retrieved_evidence: List[Evidence],
    claim_extractor: Optional[Callable] = None
) -> Tuple[float, List[str]]:
    """Detect claims in the decision that are not grounded in evidence.

    Hallucinations are statements that:
    1. Assert facts not present in retrieved evidence
    2. Cite non-existent sources
    3. Make confident claims about uncertain information

    Args:
        decision: The agent's decision
        retrieved_evidence: Evidence actually retrieved
        claim_extractor: Function to extract individual claims from justification

    Returns:
        (hallucination_rate, list_of_hallucinated_claims)

    Note:
        This is a simplified heuristic. Production systems should use
        LLM-as-Judge or more sophisticated NLI-based detection.
    """
    if claim_extractor is None:
        # Simple sentence-based claim extraction
        def claim_extractor(text):
            sentences = re.split(r'[.!?]', text)
            return [s.strip() for s in sentences if len(s.strip()) > 10]

    claims = claim_extractor(decision.justification)
    if not claims:
        return 0.0, []

    evidence_text = " ".join(e.content.lower() for e in retrieved_evidence)
    evidence_ids = {e.evidence_id.lower() for e in retrieved_evidence}

    hallucinated = []
    for claim in claims:
        claim_lower = claim.lower()

        # Check if claim references something not in evidence
        # This is a heuristic - look for specific entity mentions
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim)
        ungrounded_entities = []

        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower not in evidence_text and entity_lower not in evidence_ids:
                # Entity mentioned but not in evidence
                ungrounded_entities.append(entity)

        if len(ungrounded_entities) > len(entities) / 2:
            # More than half of entities are ungrounded
            hallucinated.append(claim)

    hallucination_rate = len(hallucinated) / len(claims) if claims else 0.0
    return hallucination_rate, hallucinated


def compute_justification_validity(
    decision: AgentDecision,
    retrieved_evidence: List[Evidence],
    reasoning_trace: Optional[List[str]] = None
) -> float:
    """Measure how well the justification traces back to evidence.

    A valid justification:
    1. References evidence that was actually retrieved
    2. Draws conclusions that follow from the evidence
    3. Acknowledges uncertainty when evidence is weak

    Args:
        decision: The agent's decision
        retrieved_evidence: Evidence actually retrieved
        reasoning_trace: Optional intermediate reasoning steps

    Returns:
        Validity score in [0, 1]
    """
    if not decision.justification:
        return 0.0

    # Component 1: Citation coverage
    evidence_ids_mentioned = set()
    justification_lower = decision.justification.lower()

    for evidence in retrieved_evidence:
        if evidence.evidence_id.lower() in justification_lower:
            evidence_ids_mentioned.add(evidence.evidence_id)
        # Also check for source mentions
        if evidence.source.lower() in justification_lower:
            evidence_ids_mentioned.add(evidence.evidence_id)

    citation_coverage = len(evidence_ids_mentioned) / len(retrieved_evidence) if retrieved_evidence else 0

    # Component 2: Evidence alignment (reuse semantic similarity)
    alignment = compute_evidence_alignment(decision, retrieved_evidence)

    # Component 3: Uncertainty acknowledgment
    uncertainty_phrases = [
        "may", "might", "could", "possibly", "likely", "uncertain",
        "based on available evidence", "suggests", "indicates"
    ]
    uncertainty_score = sum(1 for phrase in uncertainty_phrases
                           if phrase in justification_lower) / len(uncertainty_phrases)

    # Combine components
    validity = 0.4 * citation_coverage + 0.4 * alignment + 0.2 * uncertainty_score
    return validity


def compute_faithfulness(
    decision: AgentDecision,
    retrieved_evidence: List[Evidence],
    constraints: List[PolicyConstraint],
    context: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> FaithfulnessMetrics:
    """Compute comprehensive faithfulness metrics for an agent decision.

    Args:
        decision: The agent's decision
        retrieved_evidence: Evidence retrieved during trajectory
        constraints: Policy constraints to check
        context: Additional context for constraint evaluation
        weights: Optional custom weights for combining scores

    Returns:
        FaithfulnessMetrics with all component scores

    Example:
        >>> decision = AgentDecision("d1", {"action": "escalate"},
        ...     "Based on SEC Rule 10b-5 and precedent 2023-001...",
        ...     cited_evidence=["SEC-10b5", "precedent-2023-001"])
        >>> evidence = [Evidence("SEC-10b5", "rules_db", "Insider trading prohibition...")]
        >>> constraints = [PolicyConstraint("audit_trail", "Must have audit trail",
        ...                                 lambda a, c: 'audit' in str(a))]
        >>> metrics = compute_faithfulness(decision, evidence, constraints, {})
        >>> print(metrics.summary())
    """
    if weights is None:
        weights = {
            "grounding": 0.25,
            "alignment": 0.25,
            "constraints": 0.30,
            "validity": 0.20
        }

    # Compute individual components
    grounding_score, grounding_details = compute_evidence_grounding(decision, retrieved_evidence)
    alignment_score = compute_evidence_alignment(decision, retrieved_evidence)
    constraint_rate, constraint_details = check_constraint_satisfaction(decision, constraints, context)
    validity_score = compute_justification_validity(decision, retrieved_evidence)
    hallucination_rate, hallucinated_claims = detect_hallucinations(decision, retrieved_evidence)

    # Compute overall faithfulness
    overall = (
        weights["grounding"] * grounding_score +
        weights["alignment"] * alignment_score +
        weights["constraints"] * constraint_rate +
        weights["validity"] * validity_score
    )

    # Penalize for hallucinations
    overall *= (1 - 0.5 * hallucination_rate)

    return FaithfulnessMetrics(
        evidence_grounding_score=grounding_score,
        evidence_alignment_score=alignment_score,
        constraint_satisfaction_rate=constraint_rate,
        justification_validity_score=validity_score,
        hallucination_rate=hallucination_rate,
        overall_faithfulness=overall,
        grounding_details=grounding_details,
        constraint_details=constraint_details,
        hallucinated_claims=hallucinated_claims
    )


# ==============================================================================
# Determinism-Faithfulness Frontier Analysis
# ==============================================================================

@dataclass
class DeterminismFaithfulnessPoint:
    """A single point on the determinism-faithfulness frontier.

    Attributes:
        agent_type: Type of agent (unconstrained, schema-first, policy-gated)
        model: Model used
        determinism_score: Trajectory determinism (0-1)
        faithfulness_score: Overall faithfulness (0-1)
        config: Configuration details
    """
    agent_type: str
    model: str
    determinism_score: float
    faithfulness_score: float
    config: Dict[str, Any] = field(default_factory=dict)


def compute_frontier(
    points: List[DeterminismFaithfulnessPoint]
) -> List[DeterminismFaithfulnessPoint]:
    """Compute Pareto frontier of determinism-faithfulness trade-off.

    A point is on the frontier if no other point dominates it
    (higher on both dimensions).

    Args:
        points: List of (determinism, faithfulness) measurements

    Returns:
        List of points on the Pareto frontier
    """
    frontier = []

    for point in points:
        is_dominated = False
        for other in points:
            if (other.determinism_score > point.determinism_score and
                other.faithfulness_score > point.faithfulness_score):
                is_dominated = True
                break

        if not is_dominated:
            frontier.append(point)

    # Sort by determinism for plotting
    frontier.sort(key=lambda p: p.determinism_score)
    return frontier


def analyze_frontier(
    points: List[DeterminismFaithfulnessPoint]
) -> Dict[str, Any]:
    """Analyze the determinism-faithfulness frontier.

    Args:
        points: All measured points

    Returns:
        Analysis results including frontier points and recommendations
    """
    frontier = compute_frontier(points)

    # Group by agent type
    by_type = {}
    for p in points:
        if p.agent_type not in by_type:
            by_type[p.agent_type] = []
        by_type[p.agent_type].append(p)

    # Compute averages per type
    type_averages = {}
    for agent_type, type_points in by_type.items():
        type_averages[agent_type] = {
            "mean_determinism": np.mean([p.determinism_score for p in type_points]),
            "mean_faithfulness": np.mean([p.faithfulness_score for p in type_points]),
            "n_points": len(type_points),
            "n_on_frontier": sum(1 for p in type_points if p in frontier)
        }

    # Find best agent type for each scenario
    recommendations = {}

    # Scenario 1: Audit-critical (high faithfulness required)
    high_faith_points = [p for p in points if p.faithfulness_score >= 0.9]
    if high_faith_points:
        best_determinism = max(high_faith_points, key=lambda p: p.determinism_score)
        recommendations["audit_critical"] = best_determinism.agent_type

    # Scenario 2: High-frequency (high determinism required)
    high_det_points = [p for p in points if p.determinism_score >= 0.9]
    if high_det_points:
        best_faithfulness = max(high_det_points, key=lambda p: p.faithfulness_score)
        recommendations["high_frequency"] = best_faithfulness.agent_type

    # Scenario 3: Balanced
    balanced_points = [p for p in points
                      if p.determinism_score >= 0.7 and p.faithfulness_score >= 0.7]
    if balanced_points:
        # Maximize product (geometric mean of both)
        best_balanced = max(balanced_points,
                           key=lambda p: p.determinism_score * p.faithfulness_score)
        recommendations["balanced"] = best_balanced.agent_type

    return {
        "frontier_points": frontier,
        "type_averages": type_averages,
        "recommendations": recommendations,
        "total_points": len(points),
        "frontier_size": len(frontier)
    }


# ==============================================================================
# Example: Portfolio Constraint Agent Faithfulness
# ==============================================================================

def example_portfolio_agent_faithfulness():
    """
    Demonstrates faithfulness analysis for a portfolio constraint agent.

    Task: Propose rebalancing trade subject to constraints
    Evidence: Market data, risk model output, constraint definitions
    Constraints: Position limits, sector exposure, ESG thresholds
    """
    print("="*60)
    print("Faithfulness Analysis: Portfolio Constraint Agent")
    print("="*60)
    print()

    # Define constraints
    def position_limit_check(action, ctx):
        """Check position doesn't exceed limit."""
        return action.get('position_size', 0) <= ctx.get('max_position', 1000000)

    def sector_exposure_check(action, ctx):
        """Check sector exposure within bounds."""
        sector = action.get('sector', 'unknown')
        exposure = action.get('sector_exposure', 0)
        limit = ctx.get('sector_limits', {}).get(sector, 0.25)
        return exposure <= limit

    def esg_threshold_check(action, ctx):
        """Check ESG score meets minimum."""
        return action.get('esg_score', 0) >= ctx.get('min_esg', 50)

    constraints = [
        PolicyConstraint("position_limit", "Max position size", position_limit_check, "hard", "risk"),
        PolicyConstraint("sector_exposure", "Sector concentration limit", sector_exposure_check, "hard", "risk"),
        PolicyConstraint("esg_threshold", "Minimum ESG score", esg_threshold_check, "soft", "compliance")
    ]

    context = {
        "max_position": 500000,
        "sector_limits": {"tech": 0.30, "finance": 0.25, "energy": 0.20},
        "min_esg": 60
    }

    # Simulate retrieved evidence
    evidence = [
        Evidence("market_data_001", "market_feed",
                 "AAPL current price $175.50, 30-day volatility 22%, sector: tech"),
        Evidence("risk_model_001", "risk_engine",
                 "Portfolio VaR at 95%: $125,000, sector tech exposure: 28%"),
        Evidence("constraint_001", "policy_db",
                 "Maximum single position: $500,000, max tech sector: 30%"),
        Evidence("esg_score_001", "esg_provider",
                 "AAPL ESG score: 72/100, environmental: 68, social: 75, governance: 73")
    ]

    # Good decision (faithful)
    good_decision = AgentDecision(
        decision_id="d001",
        action={
            "trade": "buy",
            "symbol": "AAPL",
            "position_size": 300000,
            "sector": "tech",
            "sector_exposure": 0.28,
            "esg_score": 72
        },
        justification="""
        Based on market data (market_data_001), AAPL is trading at $175.50 with
        moderate volatility. Risk model (risk_model_001) shows current tech sector
        exposure at 28%, below the 30% limit defined in constraint_001.
        The position size of $300,000 is within the $500,000 maximum.
        ESG score of 72 from esg_score_001 exceeds the minimum threshold.
        Recommendation: BUY with stated position sizing.
        """,
        cited_evidence=["market_data_001", "risk_model_001", "constraint_001", "esg_score_001"],
        confidence=0.85
    )

    print("GOOD DECISION (Faithful):")
    print("-"*60)
    good_metrics = compute_faithfulness(good_decision, evidence, constraints, context)
    print(good_metrics.summary())

    # Bad decision (unfaithful - hallucinations + constraint violation)
    bad_decision = AgentDecision(
        decision_id="d002",
        action={
            "trade": "buy",
            "symbol": "AAPL",
            "position_size": 750000,  # Exceeds limit!
            "sector": "tech",
            "sector_exposure": 0.28,
            "esg_score": 72
        },
        justification="""
        Based on Goldman Sachs analyst report and Bloomberg Terminal data,
        AAPL is undervalued. The Federal Reserve's latest statement suggests
        tech stocks will rally. Our proprietary momentum indicator confirms
        strong buy signal. Position sizing based on Kelly criterion optimization.
        """,
        cited_evidence=["goldman_report", "bloomberg_data", "fed_statement"],  # Not retrieved!
        confidence=0.95
    )

    print("\nBAD DECISION (Unfaithful):")
    print("-"*60)
    bad_metrics = compute_faithfulness(bad_decision, evidence, constraints, context)
    print(bad_metrics.summary())

    # Determinism-Faithfulness frontier example
    print("\n" + "="*60)
    print("DETERMINISM-FAITHFULNESS FRONTIER")
    print("="*60)

    frontier_points = [
        DeterminismFaithfulnessPoint("unconstrained", "gpt-4", 0.45, 0.55),
        DeterminismFaithfulnessPoint("unconstrained", "claude-opus", 0.50, 0.60),
        DeterminismFaithfulnessPoint("schema-first", "gpt-4", 0.85, 0.70),
        DeterminismFaithfulnessPoint("schema-first", "claude-opus", 0.90, 0.75),
        DeterminismFaithfulnessPoint("policy-gated", "gpt-4", 0.75, 0.92),
        DeterminismFaithfulnessPoint("policy-gated", "claude-opus", 0.80, 0.95),
    ]

    analysis = analyze_frontier(frontier_points)

    print(f"\nTotal configurations tested: {analysis['total_points']}")
    print(f"Points on Pareto frontier: {analysis['frontier_size']}")

    print("\nAgent Type Averages:")
    for agent_type, stats in analysis['type_averages'].items():
        print(f"  {agent_type}:")
        print(f"    Mean Determinism: {stats['mean_determinism']:.1%}")
        print(f"    Mean Faithfulness: {stats['mean_faithfulness']:.1%}")
        print(f"    On Frontier: {stats['n_on_frontier']}/{stats['n_points']}")

    print("\nRecommendations:")
    for scenario, agent_type in analysis['recommendations'].items():
        print(f"  {scenario}: {agent_type}")


if __name__ == "__main__":
    example_portfolio_agent_faithfulness()
