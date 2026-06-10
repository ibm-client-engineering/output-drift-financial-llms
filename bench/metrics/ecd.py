"""Evidence Contact Divergence (ECD) metric.

Measures whether replay runs on identical input consult different evidence
subsets. This is an observable behavioral metric that does NOT depend on
hidden reasoning text.

Definition:
    For N replay runs on identical input, let each run produce an evidence
    contact set (document IDs, chunk IDs, citations, tool-return identifiers).

    ECD = mean pairwise Jaccard distance across all runs
        JaccardDistance(A, B) = 1 - |A ∩ B| / |A ∪ B|

    When decisions are available:
        SCDE = DAR * ECD
        where DAR = count(modal_decision) / N

    Interpretation:
        High DAR + high ECD = same conclusion, different evidence.
        This indicates outcome agreement masking evidence-path instability.

Example:
    >>> from bench.metrics.ecd import compute_ecd
    >>> result = compute_ecd(
    ...     [{"doc_a", "doc_b"}, {"doc_a", "doc_c"}, {"doc_a", "doc_b"}],
    ...     decisions=["escalate", "escalate", "escalate"],
    ... )
    >>> result.ecd   # Mean pairwise Jaccard distance
    >>> result.scde  # DAR * ECD
"""

from collections import Counter
from dataclasses import dataclass
from typing import Collection, List, Optional, Set, Union

import numpy as np


@dataclass(frozen=True)
class ECDResult:
    """Result of Evidence Contact Divergence computation.

    Attributes:
        ecd: Mean pairwise Jaccard distance across all runs.
        dar: Decision Agreement Rate (if decisions provided).
        scde: DAR * ECD compound metric (if decisions provided).
        mean_contact_count: Mean evidence contacts per run.
        union_contact_count: Size of the union of all evidence contacts.
        intersection_contact_count: Size of the intersection of all contacts.
        n_runs: Number of replay runs.
        empty_evidence_warning: True if any run had zero evidence contacts.
    """
    ecd: float
    dar: Optional[float]
    scde: Optional[float]
    mean_contact_count: float
    union_contact_count: int
    intersection_contact_count: int
    n_runs: int
    empty_evidence_warning: bool


def _to_frozenset(evidence: Union[Set[str], Collection[str]]) -> frozenset:
    """Canonicalize evidence contacts to a deduplicated frozenset."""
    return frozenset(str(e).strip() for e in evidence if e)


def _jaccard_distance(a: frozenset, b: frozenset) -> float:
    """Compute Jaccard distance between two sets.

    JaccardDistance(A, B) = 1 - |A ∩ B| / |A ∪ B|
    Returns 0.0 when both sets are empty (no divergence measurable).
    """
    if not a and not b:
        return 0.0
    union_size = len(a | b)
    if union_size == 0:
        return 0.0
    return 1.0 - len(a & b) / union_size


def compute_ecd(
    evidence_sets: List[Union[Set[str], Collection[str]]],
    decisions: Optional[List[str]] = None,
) -> ECDResult:
    """Compute Evidence Contact Divergence for a set of replay runs.

    Args:
        evidence_sets: List of evidence contact sets, one per run.
            Each element is a set (or list) of canonical string IDs.
            Duplicates within a run are deduplicated before scoring.
        decisions: Optional list of decision labels, one per run.
            When provided, DAR and SCDE are computed.

    Returns:
        ECDResult with all computed fields.

    Raises:
        ValueError: If evidence_sets is empty.
        ValueError: If decisions length does not match evidence_sets length.
    """
    if not evidence_sets:
        raise ValueError("evidence_sets must be non-empty")

    if decisions is not None and len(decisions) != len(evidence_sets):
        raise ValueError(
            f"decisions length ({len(decisions)}) must match "
            f"evidence_sets length ({len(evidence_sets)})"
        )

    n = len(evidence_sets)

    # Canonicalize all evidence sets
    canonical = [_to_frozenset(es) for es in evidence_sets]

    # Check for empty evidence
    empty_warning = any(len(cs) == 0 for cs in canonical)

    # N=1 edge case
    if n == 1:
        contact_count = len(canonical[0])
        dar = 1.0 if decisions else None
        return ECDResult(
            ecd=0.0,
            dar=dar,
            scde=0.0 if decisions else None,
            mean_contact_count=float(contact_count),
            union_contact_count=contact_count,
            intersection_contact_count=contact_count,
            n_runs=1,
            empty_evidence_warning=empty_warning,
        )

    # Compute pairwise Jaccard distances
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(_jaccard_distance(canonical[i], canonical[j]))

    ecd = float(np.mean(distances)) if distances else 0.0

    # Aggregate contact stats
    all_union = frozenset().union(*canonical)
    all_intersection = canonical[0]
    for cs in canonical[1:]:
        all_intersection = all_intersection & cs

    contact_counts = [len(cs) for cs in canonical]
    mean_contact_count = float(np.mean(contact_counts))

    # Decision Agreement Rate (if decisions provided)
    dar = None
    scde = None
    if decisions is not None:
        normalized = [d.strip().lower() for d in decisions]
        counts = Counter(normalized)
        modal_count = counts.most_common(1)[0][1]
        dar = modal_count / n
        scde = dar * ecd

    return ECDResult(
        ecd=ecd,
        dar=dar,
        scde=scde,
        mean_contact_count=mean_contact_count,
        union_contact_count=len(all_union),
        intersection_contact_count=len(all_intersection),
        n_runs=n,
        empty_evidence_warning=empty_warning,
    )
