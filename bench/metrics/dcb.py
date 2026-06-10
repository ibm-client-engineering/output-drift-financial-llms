"""Decision Concentration Bias (DCB) metric.

Measures whether a model collapses to a narrow subset of decision categories,
indicating pattern-matching behavior rather than evidence-based reasoning.

Definition:
    DCB = 1 - H(p) / log(K)

    where H(p) = -sum(p_i * log(p_i)) is Shannon entropy over the decision
    distribution and K is the number of possible decision categories from
    the task ontology.

    DCB in [0, 1]:
        0 = uniform distribution across K categories
        1 = all decisions concentrated on a single category

Example:
    >>> from bench.metrics.dcb import compute_dcb
    >>> result = compute_dcb(["escalate"] * 100, benchmark="compliance")
    >>> result.dcb  # ~1.0 (all mass on single decision)
    1.0
"""

import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from bench.spec.taxonomy import get_k


@dataclass(frozen=True)
class DCBResult:
    """Result of Decision Concentration Bias computation.

    Attributes:
        dcb: Decision Concentration Bias score in [0, 1].
        entropy: Shannon entropy H(p) of the observed decision distribution.
        max_entropy: Maximum possible entropy log(K).
        distribution: Observed frequency distribution {decision: proportion}.
        n_decisions: Total number of decisions analyzed.
        k_categories: Number of possible decision categories (K).
        ontology_violation: True when more distinct labels were observed than
            the ontology permits (len(distribution) > K). In that case
            H(p) can exceed log(K) and the raw score would be negative;
            it is clamped to 0.0 and this flag is set instead of failing
            silently.
    """
    dcb: float
    entropy: float
    max_entropy: float
    distribution: Dict[str, float]
    n_decisions: int
    k_categories: int
    ontology_violation: bool = False


def compute_dcb(
    decisions: List[str],
    k: Optional[int] = None,
    benchmark: Optional[str] = None,
) -> DCBResult:
    """Compute Decision Concentration Bias for a list of decisions.

    Args:
        decisions: List of decision labels (e.g., ["escalate", "dismiss", ...]).
        k: Number of possible categories. If None, derived from benchmark
           ontology or inferred from unique decisions observed.
        benchmark: Benchmark task name (e.g., "compliance"). Used to look up
                   K from the task ontology when k is not provided.

    Returns:
        DCBResult with all computed fields.

    Raises:
        ValueError: If decisions list is empty.
    """
    if not decisions:
        raise ValueError("decisions list must be non-empty")

    # Normalize decision labels
    normalized = [d.strip().lower() for d in decisions]
    n = len(normalized)

    # Determine K
    if k is not None:
        k_categories = k
    elif benchmark is not None:
        try:
            k_categories = get_k(benchmark)
        except KeyError:
            k_categories = len(set(normalized))
    else:
        k_categories = len(set(normalized))

    # Edge case: K=1
    if k_categories <= 1:
        counts = Counter(normalized)
        distribution = {label: count / n for label, count in counts.items()}
        return DCBResult(
            dcb=1.0,
            entropy=0.0,
            max_entropy=0.0,
            distribution=distribution,
            n_decisions=n,
            k_categories=max(k_categories, 1),
        )

    # Compute frequency distribution
    counts = Counter(normalized)
    distribution = {label: count / n for label, count in counts.items()}

    # Shannon entropy: H(p) = -sum(p_i * log(p_i)) for p_i > 0
    probabilities = np.array(list(distribution.values()))
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))

    # Max entropy: log(K)
    max_entropy = float(np.log(k_categories))

    # DCB = 1 - H(p) / log(K)
    dcb = 1.0 - entropy / max_entropy

    # Detect ontology violations explicitly: more observed labels than K
    # means H(p) can exceed log(K) and the clamp below would otherwise mask
    # out-of-ontology decisions (e.g. unparsed or free-text labels).
    ontology_violation = len(distribution) > k_categories
    if ontology_violation:
        warnings.warn(
            f"DCB: observed {len(distribution)} distinct decision labels but "
            f"ontology K={k_categories}; decisions outside the task ontology "
            f"are present and dcb is clamped to [0, 1].",
            stacklevel=2,
        )

    # Clamp to [0, 1] for numerical safety
    dcb = float(np.clip(dcb, 0.0, 1.0))

    return DCBResult(
        dcb=dcb,
        entropy=entropy,
        max_entropy=max_entropy,
        distribution=distribution,
        n_decisions=n,
        k_categories=k_categories,
        ontology_violation=ontology_violation,
    )
