"""Same Conclusion, Different Reasoning (SCDR) metric.

Umbrella metric for measuring divergence in agent behavior when decisions
agree. Supports multiple divergence backends (channels):

    - trajectory: encode tool-call sequences as text, embed, compute
      pairwise cosine distance. Works with existing 4,700+ replay runs.
    - rationale: embed full reasoning text directly. Only when captured.
    - auto: use rationale if present, else trajectory.

Definition:
    DAR (Decision Agreement Rate) = count(modal_decision) / N
    RDS (Reasoning Divergence Score) = mean pairwise (1 - cosine_sim)
    SCDR = DAR * RDS

    High SCDR = agents converge on same decision through divergent
    reasoning/trajectory paths.

Example:
    >>> from bench.metrics.scdr import compute_scdr
    >>> runs = [("escalate", "check_sanctions get_profile"),
    ...         ("escalate", "get_profile check_sanctions calculate_risk")]
    >>> result = compute_scdr(runs, mode="trajectory")
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SCDRResult:
    """Result of SCDR computation.

    Attributes:
        dar: Decision Agreement Rate — proportion agreeing with modal decision.
        rds: Reasoning Divergence Score — mean pairwise divergence.
        scdr: DAR * RDS — the compound metric.
        modal_decision: Most common decision label.
        n_runs: Number of replay runs.
        decision_counts: Frequency of each decision label.
        mode: Divergence backend used ("trajectory", "rationale", "precomputed").
    """
    dar: float
    rds: float
    scdr: float
    modal_decision: str
    n_runs: int
    decision_counts: Dict[str, int]
    mode: str


def _compute_dar(decisions: List[str]) -> Tuple[float, str, Dict[str, int]]:
    """Compute Decision Agreement Rate.

    Returns (dar, modal_decision, decision_counts).
    """
    normalized = [d.strip().lower() for d in decisions]
    counts = Counter(normalized)
    modal_decision, modal_count = counts.most_common(1)[0]
    dar = modal_count / len(normalized)
    return dar, modal_decision, dict(counts)


def _tool_sequence_to_text(tool_sequence: str) -> str:
    """Normalize a tool sequence string for embedding.

    Ensures consistent text representation for cosine similarity.
    """
    return tool_sequence.strip().lower()


def _pairwise_cosine_distances(embeddings: np.ndarray) -> float:
    """Compute mean pairwise (1 - cosine_similarity) from an embedding matrix.

    Args:
        embeddings: (N, D) array of normalized embeddings.

    Returns:
        Mean pairwise cosine distance. 0.0 if N < 2.
    """
    n = embeddings.shape[0]
    if n < 2:
        return 0.0

    # Normalize rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid div by zero
    normed = embeddings / norms

    # Cosine similarity matrix
    sim_matrix = normed @ normed.T

    # Extract upper triangle (excluding diagonal)
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_indices]

    # Mean cosine distance
    return float(np.mean(1.0 - pairwise_sims))


def _embed_traces(traces: List[str], model_name: str) -> np.ndarray:
    """Embed text traces using sentence-transformers.

    Lazy-loads the model to avoid import-time overhead.

    Args:
        traces: List of text strings to embed.
        model_name: sentence-transformers model name.

    Returns:
        (N, D) numpy array of embeddings.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for SCDR computation. "
            "Install with: pip install sentence-transformers"
        )

    # Module-level model cache
    if not hasattr(_embed_traces, "_model_cache"):
        _embed_traces._model_cache = {}

    if model_name not in _embed_traces._model_cache:
        _embed_traces._model_cache[model_name] = SentenceTransformer(model_name)

    model = _embed_traces._model_cache[model_name]
    return model.encode(traces, convert_to_numpy=True, show_progress_bar=False)


def compute_scdr(
    runs: List[Tuple[str, str]],
    mode: str = "auto",
    model_name: str = "all-MiniLM-L6-v2",
    embeddings: Optional[np.ndarray] = None,
) -> SCDRResult:
    """Compute Same Conclusion, Different Reasoning metric.

    Args:
        runs: List of (decision, trace) tuples. The trace is either a
            tool sequence string (trajectory mode) or reasoning text
            (rationale mode).
        mode: Divergence backend — "trajectory", "rationale", or "auto".
            In "auto" mode, traces with >100 chars are treated as rationale,
            otherwise as trajectory.
        model_name: sentence-transformers model for embedding. Ignored if
            embeddings are provided.
        embeddings: Pre-computed (N, D) embedding array. When provided,
            skips model loading and encoding entirely.

    Returns:
        SCDRResult with all computed fields.

    Raises:
        ValueError: If runs is empty.
    """
    if not runs:
        raise ValueError("runs list must be non-empty")

    decisions = [r[0] for r in runs]
    traces = [r[1] for r in runs]
    n = len(runs)

    # N=1 edge case
    if n == 1:
        dar, modal, counts = _compute_dar(decisions)
        return SCDRResult(
            dar=dar, rds=0.0, scdr=0.0,
            modal_decision=modal, n_runs=1,
            decision_counts=counts, mode=mode,
        )

    # Determine actual mode
    actual_mode = mode
    if mode == "auto":
        avg_len = sum(len(t) for t in traces) / max(len(traces), 1)
        actual_mode = "rationale" if avg_len > 100 else "trajectory"

    # Compute DAR
    dar, modal_decision, decision_counts = _compute_dar(decisions)

    # Compute RDS
    if embeddings is not None:
        rds = _pairwise_cosine_distances(embeddings)
        actual_mode = "precomputed"
    else:
        # Prepare traces for embedding
        if actual_mode == "trajectory":
            processed = [_tool_sequence_to_text(t) for t in traces]
        else:
            processed = traces

        # Check for all-identical traces
        if len(set(processed)) == 1:
            rds = 0.0
        else:
            emb = _embed_traces(processed, model_name)
            rds = _pairwise_cosine_distances(emb)

    scdr = dar * rds

    return SCDRResult(
        dar=dar, rds=rds, scdr=scdr,
        modal_decision=modal_decision, n_runs=n,
        decision_counts=decision_counts, mode=actual_mode,
    )
