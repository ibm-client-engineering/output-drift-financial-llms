"""Deterministic finite-corpus sensitivity utilities."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ..models import Trajectory
from .agreement import dar, tar


@dataclass(frozen=True)
class IntervalResult:
    """Point and percentile range from finite-corpus case resampling."""

    point: float
    low: float
    high: float
    draws: int
    level: float
    seed: int


@dataclass(frozen=True)
class PermutationResult:
    """Two-sided paired sign-flip result."""

    statistic: float
    p_value: float
    permutations: int
    seed: int


@dataclass(frozen=True)
class SubsampleResult:
    """Distribution summary after replay-count matching."""

    dar_median: float
    dar_low: float
    dar_high: float
    tar_median: float
    tar_low: float
    tar_high: float
    gap_median: float
    gap_low: float
    gap_high: float
    draws: int
    k: int
    seed: int


@dataclass(frozen=True)
class LOOResult:
    """Leave-one-case-out range."""

    full: float
    minimum: float
    maximum: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class BoundPoint:
    """One adversarial parser-fallback sensitivity point."""

    fraction: float
    episodes_flipped: int
    dar: float
    tar: float
    gap: float


@dataclass(frozen=True)
class BoundResult:
    """Worst-case modal-label reassignment over a declared fraction grid."""

    baseline_dar: float
    baseline_tar: float
    baseline_gap: float
    points: tuple[BoundPoint, ...]
    first_sign_change_fraction: float | None


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def case_resampling_interval(
    values: Sequence[float], *, n: int = 5000, seed: int = 42, level: float = 0.95
) -> IntervalResult:
    """Resample retained cases with replacement.

    This is a finite-corpus sensitivity range, not a population confidence
    interval. It does not repair non-random case or provider selection.
    """

    if not values or n < 1 or not 0.0 < level < 1.0:
        raise ValueError("values, n, and level are invalid")
    rng = random.Random(seed)
    draws = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(n)]
    alpha = (1.0 - level) / 2.0
    return IntervalResult(
        point=sum(values) / len(values),
        low=_quantile(draws, alpha),
        high=_quantile(draws, 1.0 - alpha),
        draws=n,
        level=level,
        seed=seed,
    )


def sign_flip_permutation(
    gaps: Sequence[float], *, n: int = 10000, seed: int = 42
) -> PermutationResult:
    """Within-pair sign-flip test; it assumes exchangeability of gap signs."""

    if not gaps or n < 1:
        raise ValueError("gaps and n must be nonempty/positive")
    observed = sum(gaps) / len(gaps)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n):
        value = sum(gap if rng.getrandbits(1) else -gap for gap in gaps) / len(gaps)
        extreme += abs(value) >= abs(observed) - 1e-15
    return PermutationResult(observed, (extreme + 1) / (n + 1), n, seed)


def subsample_replays(
    groups: Sequence[tuple[Sequence[str], Sequence[Trajectory]]],
    *,
    k: int = 3,
    draws: int = 500,
    seed: int = 42,
) -> SubsampleResult:
    """Match replay count by independently sampling ``k`` episodes per group."""

    if not groups or k < 1 or draws < 1:
        raise ValueError("groups, k, and draws must be nonempty/positive")
    if any(len(decisions) != len(paths) or len(decisions) < k for decisions, paths in groups):
        raise ValueError("each group needs a shared denominator of at least k")
    rng = random.Random(seed)
    dar_draws: list[float] = []
    tar_draws: list[float] = []
    gap_draws: list[float] = []
    for _ in range(draws):
        case_dar: list[float] = []
        case_tar: list[float] = []
        for decisions, paths in groups:
            indices = rng.sample(range(len(decisions)), k)
            selected_decisions = [decisions[index] for index in indices]
            selected_paths = [paths[index] for index in indices]
            case_dar.append(dar(selected_decisions))
            case_tar.append(tar(selected_paths))
        dar_value = sum(case_dar) / len(case_dar)
        tar_value = sum(case_tar) / len(case_tar)
        dar_draws.append(dar_value)
        tar_draws.append(tar_value)
        gap_draws.append(dar_value - tar_value)
    return SubsampleResult(
        _quantile(dar_draws, 0.5),
        _quantile(dar_draws, 0.025),
        _quantile(dar_draws, 0.975),
        _quantile(tar_draws, 0.5),
        _quantile(tar_draws, 0.025),
        _quantile(tar_draws, 0.975),
        _quantile(gap_draws, 0.5),
        _quantile(gap_draws, 0.025),
        _quantile(gap_draws, 0.975),
        draws,
        k,
        seed,
    )


def leave_one_case_out(values: Sequence[float]) -> LOOResult:
    """Return the full mean and every leave-one-case-out mean."""

    if len(values) < 2:
        raise ValueError("leave-one-case-out requires at least two cases")
    total = sum(values)
    loo = tuple((total - value) / (len(values) - 1) for value in values)
    return LOOResult(total / len(values), min(loo), max(loo), loo)


def decision_concentration(decisions: Sequence[str], *, k_labels: int = 3) -> float:
    """Normalized concentration ``1 - H(p)/log(K)`` for screening only."""

    if not decisions or k_labels < 2:
        raise ValueError("decisions must be nonempty and k_labels >= 2")
    counts = Counter(label.strip().lower() for label in decisions)
    if len(counts) > k_labels:
        raise ValueError("observed decisions exceed the declared ontology")
    total = len(decisions)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return max(0.0, min(1.0, 1.0 - entropy / math.log(k_labels)))


def adversarial_fallback_bound(
    groups: Sequence[tuple[Sequence[str], Sequence[Trajectory]]],
    k_grid: Sequence[float],
) -> BoundResult:
    """Flip up to ``k`` of retained decisions away from their case mode.

    This is a worst-case measurement sensitivity bound, not an estimate of the
    parser's failure rate. Each candidate episode can be reassigned once. At
    every step the deterministic adversary chooses the reassignment that
    minimizes equal-case-weighted DAR; paths and the shared denominator stay
    fixed. Callers needing task weighting should run the bound within task and
    combine task summaries explicitly.
    """

    if not groups or not k_grid:
        raise ValueError("groups and k_grid must be nonempty")
    if any(not 0.0 <= fraction <= 1.0 for fraction in k_grid):
        raise ValueError("k_grid fractions must be in [0, 1]")
    if any(not decisions or len(decisions) != len(paths) for decisions, paths in groups):
        raise ValueError("each group needs a nonempty shared decision/path denominator")

    normalized = [
        [decision.strip().lower() for decision in decisions] for decisions, _ in groups
    ]
    if any(not decision for group in normalized for decision in group):
        raise ValueError("decision labels must be nonempty")
    paths = [tuple(group_paths) for _, group_paths in groups]
    total_episodes = sum(len(group) for group in normalized)
    labels = sorted({label for group in normalized for label in group})
    if len(labels) == 1:
        labels.append("__adversarial_alternative__")

    def mean_dar(values: Sequence[Sequence[str]]) -> float:
        return sum(dar(group) for group in values) / len(values)

    baseline_dar = mean_dar(normalized)
    baseline_tar = sum(tar(group_paths) for group_paths in paths) / len(paths)
    baseline_gap = baseline_dar - baseline_tar
    points: list[BoundPoint] = []
    first_sign_change: float | None = None

    for fraction in k_grid:
        decisions = [list(group) for group in normalized]
        available = {
            (group_index, episode_index)
            for group_index, group in enumerate(decisions)
            for episode_index in range(len(group))
        }
        target = min(total_episodes, math.ceil(total_episodes * fraction - 1e-15))
        applied = 0
        while applied < target and available:
            best: tuple[float, int, int, str] | None = None
            for group_index, episode_index in sorted(available):
                current = decisions[group_index][episode_index]
                counts = Counter(decisions[group_index])
                maximum = max(counts.values())
                if counts[current] != maximum:
                    continue
                for replacement in labels:
                    if replacement == current:
                        continue
                    candidate = list(decisions[group_index])
                    candidate[episode_index] = replacement
                    score = dar(candidate)
                    option = (score, group_index, episode_index, replacement)
                    if best is None or option < best:
                        best = option
            if best is None:
                break
            _, group_index, episode_index, replacement = best
            decisions[group_index][episode_index] = replacement
            available.remove((group_index, episode_index))
            applied += 1

        bounded_dar = mean_dar(decisions)
        bounded_gap = bounded_dar - baseline_tar
        point = BoundPoint(
            fraction=fraction,
            episodes_flipped=applied,
            dar=bounded_dar,
            tar=baseline_tar,
            gap=bounded_gap,
        )
        points.append(point)
        sign_changed = (baseline_gap > 0.0 and bounded_gap <= 0.0) or (
            baseline_gap < 0.0 and bounded_gap >= 0.0
        )
        if sign_changed and first_sign_change is None:
            first_sign_change = fraction

    return BoundResult(
        baseline_dar=baseline_dar,
        baseline_tar=baseline_tar,
        baseline_gap=baseline_gap,
        points=tuple(points),
        first_sign_change_fraction=first_sign_change,
    )
