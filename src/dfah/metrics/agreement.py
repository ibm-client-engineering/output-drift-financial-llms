"""Decision and trajectory agreement metrics."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from ..models import (
    CaseReport,
    Eligibility,
    EligibilityReasonCount,
    Episode,
    TARReport,
    ToolCall,
    Trajectory,
)
from .eligibility import evaluate_group

_T = TypeVar("_T", bound=Hashable)


class PathMode(str, Enum):
    """Trajectory abstraction used for modal agreement."""

    SEQ = "seq"
    BAG = "bag"
    SET = "set"
    STRONG = "strong"


@dataclass(frozen=True)
class AgreementResult:
    """Modal agreement plus explicit ties and denominator."""

    value: float
    denominator: int
    modes: tuple[Hashable, ...]
    tied: bool


@dataclass(frozen=True)
class ConfigReport:
    """Task-balanced summary across case reports."""

    dar: float
    tar: TARReport
    gap: float
    n_tasks: int
    n_cases: int


@dataclass(frozen=True)
class ReplayGroupEvaluation:
    """Eligibility result and episodes for one expected replay group."""

    task: str
    case_id: str
    episodes: tuple[Episode, ...]
    eligibility: Eligibility


def modal_agreement(values: Sequence[_T]) -> AgreementResult:
    """Return the largest observed frequency divided by ``N``.

    Ties are represented explicitly; insertion order never selects a hidden
    canonical mode.
    """

    if not values:
        raise ValueError("modal agreement requires at least one value")
    counts = Counter(values)
    maximum = max(counts.values())
    modes = tuple(
        sorted((value for value, count in counts.items() if count == maximum), key=repr)
    )
    return AgreementResult(maximum / len(values), len(values), modes, len(modes) > 1)


def first_anchored_agreement(values: Sequence[_T]) -> float:
    """Fraction matching the first replay; never exceeds modal agreement."""

    if not values:
        raise ValueError("first-anchored agreement requires at least one value")
    anchor = values[0]
    return sum(value == anchor for value in values) / len(values)


def dar(decisions: Sequence[str]) -> float:
    """Decision agreement rate: modal decision frequency divided by ``N``."""

    normalized = [decision.strip().lower() for decision in decisions]
    if any(not value for value in normalized):
        raise ValueError("decision labels must be nonempty")
    return modal_agreement(normalized).value


def _sequence(path: Trajectory | Sequence[ToolCall]) -> tuple[str, ...]:
    calls = path.tool_calls if isinstance(path, Trajectory) else tuple(path)
    return tuple(call.name for call in calls)


def _path_key(path: Trajectory | Sequence[ToolCall], mode: PathMode) -> Hashable:
    calls = path.tool_calls if isinstance(path, Trajectory) else tuple(path)
    names = tuple(call.name for call in calls)
    if mode is PathMode.SEQ:
        return names
    if mode is PathMode.BAG:
        return tuple(sorted(Counter(names).items()))
    if mode is PathMode.SET:
        return frozenset(names)
    if mode is PathMode.STRONG:
        return tuple(
            (
                call.name,
                call.argument_hash,
                call.output_hash,
                call.result_state.value,
                call.execution_state.value,
            )
            for call in calls
        )
    raise ValueError(f"unsupported path mode: {mode}")


def tar(
    paths: Sequence[Trajectory | Sequence[ToolCall]],
    mode: PathMode | str = PathMode.SEQ,
) -> float:
    """Trajectory agreement rate under sequence, bag, set, or strong identity."""

    selected = PathMode(mode)
    return modal_agreement([_path_key(path, selected) for path in paths]).value


def delta_dt(decisions: Sequence[str], paths: Sequence[Trajectory]) -> float:
    """Paired within-group decision minus sequence-path agreement."""

    if len(decisions) != len(paths):
        raise ValueError("DAR and TAR require an identical episode denominator")
    return dar(decisions) - tar(paths, PathMode.SEQ)


def tar_report(paths: Sequence[Trajectory]) -> TARReport:
    """Compute every supported path abstraction on one shared denominator."""

    return TARReport(
        seq=tar(paths, PathMode.SEQ),
        bag=tar(paths, PathMode.BAG),
        set=tar(paths, PathMode.SET),
        strong=tar(paths, PathMode.STRONG),
    )


def _replay_group_evaluations(
    episodes: Sequence[Episode],
    *,
    required_replays: int,
    expected_groups: Sequence[tuple[str, str]] = (),
) -> tuple[ReplayGroupEvaluation, ...]:
    groups: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for episode in episodes:
        groups[(episode.task, episode.case_id)].append(episode)
    identities = set(groups)
    identities.update(expected_groups)
    evaluations: list[ReplayGroupEvaluation] = []
    for task, case_id in sorted(identities):
        rows = sorted(groups.get((task, case_id), ()), key=lambda episode: episode.replay_index)
        evaluations.append(
            ReplayGroupEvaluation(
                task=task,
                case_id=case_id,
                episodes=tuple(rows),
                eligibility=evaluate_group(rows, required_replays=required_replays),
            )
        )
    return tuple(evaluations)


def _group_reason(reason: str) -> str:
    """Remove replay ordinals so diagnostics count affected groups, not positions."""

    return re.sub(r"^replay_\d+:", "", reason)


def ineligibility_summary_from_episodes(
    episodes: Sequence[Episode],
    *,
    required_replays: int,
    expected_groups: Sequence[tuple[str, str]] = (),
) -> tuple[int, tuple[EligibilityReasonCount, ...]]:
    """Return privacy-safe group exclusions reconstructed from episode artifacts."""

    evaluations = _replay_group_evaluations(
        episodes,
        required_replays=required_replays,
        expected_groups=expected_groups,
    )
    ineligible = [
        evaluation for evaluation in evaluations if not evaluation.eligibility.eligible
    ]
    reason_counts: Counter[str] = Counter()
    for evaluation in ineligible:
        reason_counts.update(
            {_group_reason(reason) for reason in evaluation.eligibility.reasons}
        )
    return len(ineligible), tuple(
        EligibilityReasonCount(reason=reason, groups=groups)
        for reason, groups in sorted(reason_counts.items())
    )


def case_reports_from_episodes(
    episodes: Sequence[Episode], *, required_replays: int
) -> tuple[CaseReport, ...]:
    """Regenerate eligible case reports from committed episode artifacts."""

    reports: list[CaseReport] = []
    for evaluation in _replay_group_evaluations(episodes, required_replays=required_replays):
        task, case_id = evaluation.task, evaluation.case_id
        rows = evaluation.episodes
        eligibility = evaluation.eligibility
        if not eligibility.eligible:
            continue
        decisions = [episode.decision.label for episode in rows if episode.decision is not None]
        paths = [episode.trajectory for episode in rows]
        decision_agreement = dar(decisions)
        trajectory_agreement = tar_report(paths)
        reports.append(
            CaseReport(
                case_id=case_id,
                task=task,
                replay_count=len(rows),
                decision_denominator=len(rows),
                trajectory_denominator=len(rows),
                dar=decision_agreement,
                tar=trajectory_agreement,
                gap=decision_agreement - trajectory_agreement.seq,
                eligibility=eligibility,
                unanimous_with_sequence_change=(
                    decision_agreement == 1.0 and trajectory_agreement.seq < 1.0
                ),
                unanimous_with_path_change=(
                    decision_agreement == 1.0 and trajectory_agreement.strong < 1.0
                ),
            )
        )
    return tuple(reports)


def task_weighted(case_reports: Sequence[CaseReport]) -> ConfigReport:
    """Average within task, then give every represented task equal weight."""

    eligible = [report for report in case_reports if report.eligibility.eligible]
    if not eligible:
        raise ValueError("task-weighted summary requires an eligible case")
    by_task: dict[str, list[CaseReport]] = defaultdict(list)
    for report in eligible:
        by_task[report.task].append(report)

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    task_dar = [mean([row.dar for row in rows]) for rows in by_task.values()]
    task_seq = [mean([row.tar.seq for row in rows]) for rows in by_task.values()]
    task_bag = [mean([row.tar.bag for row in rows]) for rows in by_task.values()]
    task_set = [mean([row.tar.set for row in rows]) for rows in by_task.values()]
    task_strong = [mean([row.tar.strong for row in rows]) for rows in by_task.values()]
    summary_tar = TARReport(
        seq=mean(task_seq),
        bag=mean(task_bag),
        set=mean(task_set),
        strong=mean(task_strong),
    )
    summary_dar = mean(task_dar)
    return ConfigReport(
        dar=summary_dar,
        tar=summary_tar,
        gap=summary_dar - summary_tar.seq,
        n_tasks=len(by_task),
        n_cases=len(eligible),
    )
