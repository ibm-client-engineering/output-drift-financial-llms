#!/usr/bin/env python3
"""Build the corrected arXiv v2 retrospective analysis artifacts.

The corrected analysis intentionally excludes the portfolio fixture.  A July 2026
artifact audit found inconsistencies between its declared portfolio value,
runtime tool outputs, and several reference rationales.  Those episodes remain
in the repository for forensic reproducibility, but they are not used here.

Unlike the legacy analysis, an empty tool path is an observed trajectory state.
This keeps DAR and TAR on the same repeated-run denominator.  Configurations
that never called a tool are reported separately and are not interpreted as
successful tool-using agents.

Ordered tool-name sequences are the primary trajectory observable.  Two
sensitivity abstractions make the construct explicit: a multiset retains call
multiplicity while ignoring order, and a set ignores both order and
multiplicity.  None of these legacy-log measures observe tool arguments,
results, or semantic policy equivalence.

This public builder reads a channel-minimal synthetic fixture.  It does not
need provider access, prompts, tool arguments, tool results, or local paths.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "results" / "v2" / "fixtures" / "retrospective_episodes.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "v2" / "retrospective"
EVAL_TASKS = ("compliance", "dataops")
EXCLUDED_MODELS = {"deepseek-r1_8b"}
ZERO_TOOL_CONFIGURATION_EXCLUSIONS = {
    "granite3.3_latest": "historical configuration observed zero tool calls",
    "mistral_7b": "historical configuration observed zero tool calls",
}
SEED = 42
N_BOOTSTRAP = 5_000

MODEL_ORDER = [
    "qwen3.5_latest",
    "gemma4_latest",
    "qwen2.5_7b-instruct",
    "gpt-oss_20b",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
]

MODEL_NAMES = {
    "qwen3.5_latest": "Qwen 3.5",
    "gemma4_latest": "Gemma 4",
    "qwen2.5_7b-instruct": "Qwen 2.5 7B",
    "gpt-oss_20b": "GPT-OSS 20B",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "claude-opus-4-20250514": "Claude Opus 4",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
}

METRIC_COLUMNS = (
    "dar",
    "tar",
    "tar_multiset",
    "tar_set",
    "gap",
    "gap_multiset",
    "gap_set",
)

PROTOCOL = {
    "qwen3.5_latest": "temperature 0, seed 42",
    "gemma4_latest": "temperature 0, seed 42",
    "qwen2.5_7b-instruct": "temperature 0, seed 42",
    "gpt-oss_20b": "temperature 0, seed 42",
    "gemini-2.0-flash": "temperature 0, no seed control",
    "gemini-2.5-pro": "temperature 0, no seed control",
    "claude-opus-4-20250514": "provider-default temperature, no seed control",
    "claude-sonnet-4-20250514": "provider-default temperature, no seed control",
}


@dataclass(frozen=True)
class Episode:
    decision: str
    path: tuple[str, ...]
    run: int = 0


CaseKey = tuple[str, str, str]  # task, model, case_id


def agreement(values: list[object]) -> float:
    """Fraction of a repeated-run sample matching its modal value."""
    if not values:
        return float("nan")
    return Counter(values).most_common(1)[0][1] / len(values)


def modal_value(values: list[str]) -> str:
    """Return a deterministic modal value for artifact reporting."""
    counts = Counter(values)
    max_count = max(counts.values())
    return min(value for value, count in counts.items() if count == max_count)


def path_multiset(path: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    """Canonical tool-name bag: order-free while retaining call counts."""
    return tuple(sorted(Counter(path).items()))


def path_set(path: tuple[str, ...]) -> tuple[str, ...]:
    """Canonical tool-name set: order- and multiplicity-free sensitivity."""
    return tuple(sorted(set(path)))


FixtureRow = dict[str, object]


def load_fixture(fixture_path: Path) -> list[FixtureRow]:
    """Load and validate the public channel-minimal synthetic fixture."""
    required = {
        "task",
        "model",
        "case_id",
        "replay",
        "decision",
        "tool_names",
    }
    rows: list[FixtureRow] = []
    with fixture_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if set(row) != required:
                raise ValueError(f"{fixture_path}:{line_number}: fixture fields changed")
            if not all(
                isinstance(row[field], str) and row[field]
                for field in ("task", "model", "case_id")
            ):
                raise ValueError(f"{fixture_path}:{line_number}: invalid grouping key")
            if not isinstance(row["replay"], int) or isinstance(row["replay"], bool):
                raise ValueError(f"{fixture_path}:{line_number}: invalid replay index")
            rows.append(row)
    return rows


def _repeated_groups(
    fixture_rows: list[FixtureRow],
    *,
    tasks: set[str] | None = None,
    excluded_models: set[str] | None = None,
) -> dict[CaseKey, list[Episode]]:
    groups: dict[CaseKey, list[Episode]] = defaultdict(list)
    ineligible_trajectory_groups: set[CaseKey] = set()
    for row in fixture_rows:
        task = str(row["task"])
        model = str(row["model"])
        if tasks is not None and task not in tasks:
            continue
        if excluded_models is not None and model in excluded_models:
            continue
        raw_decision = row["decision"]
        decision = raw_decision if isinstance(raw_decision, str) else ""
        if not decision:
            continue
        key = (task, model, str(row["case_id"]))
        raw_tool_path = row["tool_names"]
        if not isinstance(raw_tool_path, list) or not all(
            isinstance(name, str) and name for name in raw_tool_path
        ):
            ineligible_trajectory_groups.add(key)
            continue
        tool_path = tuple(raw_tool_path)
        run_index = int(row["replay"])
        groups[key].append(Episode(decision, tool_path, run_index))
    return {
        key: episodes
        for key, episodes in groups.items()
        if key not in ineligible_trajectory_groups and len(episodes) >= 2
    }


def load_groups(fixture_rows: list[FixtureRow]) -> dict[CaseKey, list[Episode]]:
    return _repeated_groups(
        fixture_rows,
        tasks=set(EVAL_TASKS),
        excluded_models=EXCLUDED_MODELS,
    )


def corpus_lineage(fixture_rows: list[FixtureRow]) -> pd.DataFrame:
    """Reconstruct each published v2 corpus transition from the fixture."""
    raw_groups = {
        (str(row["task"]), str(row["model"]), str(row["case_id"])) for row in fixture_rows
    }
    repeated = _repeated_groups(fixture_rows)
    without_portfolio = {
        key: episodes for key, episodes in repeated.items() if key[0] in EVAL_TASKS
    }
    without_deepseek = {
        key: episodes
        for key, episodes in without_portfolio.items()
        if key[1] not in EXCLUDED_MODELS
    }
    primary = {
        key: episodes
        for key, episodes in without_deepseek.items()
        if key[1] not in ZERO_TOOL_CONFIGURATION_EXCLUSIONS
    }

    stages = [
        (
            "raw_replay_ledger",
            len(fixture_rows),
            len(raw_groups),
            "all sanitized compact episode records",
        ),
        (
            "archived_v1_after_singleton_removal",
            sum(len(episodes) for episodes in repeated.values()),
            len(repeated),
            "three-task repeated-execution corpus",
        ),
        (
            "remove_analyzed_portfolio_fixture",
            sum(len(episodes) for episodes in without_portfolio.values()),
            len(without_portfolio),
            "compliance and dataops only",
        ),
        (
            "remove_deepseek_compliance_pilot",
            sum(len(episodes) for episodes in without_deepseek.values()),
            len(without_deepseek),
            "measurement-audited retained ledger",
        ),
        (
            "retain_configurations_with_observed_tool_use",
            sum(len(episodes) for episodes in primary.values()),
            len(primary),
            "primary retrospective DAR-TAR analysis",
        ),
    ]
    rows = []
    previous_episodes = stages[0][1]
    previous_groups = stages[0][2]
    for index, (stage, episodes, groups, interpretation) in enumerate(stages):
        rows.append(
            {
                "stage": stage,
                "episodes": episodes,
                "groups": groups,
                "episode_change": 0 if index == 0 else episodes - previous_episodes,
                "group_change": 0 if index == 0 else groups - previous_groups,
                "interpretation": interpretation,
            }
        )
        previous_episodes = episodes
        previous_groups = groups
    return pd.DataFrame(rows)


def trajectory_channel_eligibility(
    fixture_rows: list[FixtureRow],
) -> pd.DataFrame:
    """Account for required trajectory-channel failures in the analysis scope."""
    invalid_episodes = 0
    empty_decision_episodes = 0
    invalid_groups: set[CaseKey] = set()
    empty_decision_groups: set[CaseKey] = set()
    inspected_episodes = 0
    for row in fixture_rows:
        task = str(row["task"])
        model = str(row["model"])
        if task not in EVAL_TASKS or model in EXCLUDED_MODELS:
            continue
        inspected_episodes += 1
        key = (task, model, str(row["case_id"]))
        decision = row["decision"] if isinstance(row["decision"], str) else ""
        if not decision:
            empty_decision_episodes += 1
            empty_decision_groups.add(key)
            continue
        raw_tool_path = row["tool_names"]
        if not isinstance(raw_tool_path, list) or not all(
            isinstance(name, str) and name for name in raw_tool_path
        ):
            invalid_episodes += 1
            invalid_groups.add(key)
    return pd.DataFrame(
        [
            {
                "scope": "compliance_and_dataops_excluding_deepseek_pilot",
                "inspected_episodes": inspected_episodes,
                "missing_decision_episodes": empty_decision_episodes,
                "missing_decision_affected_groups": len(empty_decision_groups),
                "missing_or_malformed_trajectory_episodes": invalid_episodes,
                "missing_or_malformed_trajectory_affected_groups": len(invalid_groups),
                "policy": "affected replay group is ineligible",
            }
        ]
    )


def case_frame(groups: dict[CaseKey, list[Episode]]) -> pd.DataFrame:
    rows = []
    for (task, model, case_id), episodes in groups.items():
        decisions = [episode.decision for episode in episodes]
        paths = [episode.path for episode in episodes]
        dar = agreement(decisions)
        tar = agreement(paths)
        tar_multiset = agreement([path_multiset(path) for path in paths])
        tar_set = agreement([path_set(path) for path in paths])
        rows.append(
            {
                "task": task,
                "model": model,
                "case_id": case_id,
                "n_runs": len(episodes),
                "modal_decision": modal_value(decisions),
                "dar": dar,
                "tar": tar,
                "tar_multiset": tar_multiset,
                "tar_set": tar_set,
                "gap": dar - tar,
                "gap_multiset": dar - tar_multiset,
                "gap_set": dar - tar_set,
                "tool_run_rate": np.mean([bool(path) for path in paths]),
            }
        )
    return pd.DataFrame(rows)


def task_weighted_summary(case_df: pd.DataFrame) -> pd.DataFrame:
    task = case_df.groupby(["model", "task"], as_index=False).agg(
        n_cases=("case_id", "size"),
        n_episodes=("n_runs", "sum"),
        dar=("dar", "mean"),
        tar=("tar", "mean"),
        tar_multiset=("tar_multiset", "mean"),
        tar_set=("tar_set", "mean"),
        gap=("gap", "mean"),
        gap_multiset=("gap_multiset", "mean"),
        gap_set=("gap_set", "mean"),
        tool_run_rate=("tool_run_rate", "mean"),
    )
    summary = task.groupby("model", as_index=False).agg(
        n_tasks=("task", "size"),
        n_case_groups=("n_cases", "sum"),
        n_episodes=("n_episodes", "sum"),
        dar=("dar", "mean"),
        tar=("tar", "mean"),
        tar_multiset=("tar_multiset", "mean"),
        tar_set=("tar_set", "mean"),
        gap=("gap", "mean"),
        gap_multiset=("gap_multiset", "mean"),
        gap_set=("gap_set", "mean"),
        tool_run_rate=("tool_run_rate", "mean"),
    )
    summary["model_name"] = summary["model"].map(MODEL_NAMES)
    summary["protocol"] = summary["model"].map(PROTOCOL)
    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    summary["_order"] = summary["model"].map(order)
    return summary.sort_values("_order").drop(columns="_order")


def task_summary(case_df: pd.DataFrame) -> pd.DataFrame:
    """Write task-level metrics, including reproducible decision concentration."""
    rows: list[dict[str, object]] = []
    for (model, task), group in case_df.groupby(["model", "task"], sort=True):
        distribution = Counter(group["modal_decision"])
        probabilities = np.array(list(distribution.values()), dtype=float) / len(group)
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        concentration = 1.0 - entropy / np.log(3)
        row: dict[str, object] = {
            "model": model,
            "model_name": MODEL_NAMES[model],
            "task": task,
            "n_cases": len(group),
            "n_episodes": int(group["n_runs"].sum()),
            "min_repeats": int(group["n_runs"].min()),
            "max_repeats": int(group["n_runs"].max()),
            "decision_concentration": concentration,
            "modal_decision_counts": json.dumps(dict(sorted(distribution.items()))),
        }
        for metric in METRIC_COLUMNS:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def rq2_decomposition(
    groups: dict[CaseKey, list[Episode]], case_df: pd.DataFrame
) -> pd.DataFrame:
    """Decompose sequence variation under unanimous observed decisions."""
    tool_rates = case_df.groupby("model")["tool_run_rate"].mean()
    tool_models = set(tool_rates[tool_rates > 0.5].index)
    buckets: dict[str, Counter[str]] = defaultdict(Counter)

    for (_task, model, _case_id), episodes in groups.items():
        if len({episode.decision for episode in episodes}) != 1:
            continue
        paths = [episode.path for episode in episodes]
        scopes = ["__all__", model]
        if model in tool_models:
            scopes.append("__tool_calling__")
        for scope in scopes:
            buckets[scope]["unanimous_groups"] += 1
        if len(set(paths)) == 1:
            continue

        bags = [path_multiset(path) for path in paths]
        sets = [path_set(path) for path in paths]
        if len(set(bags)) == 1:
            category = "reorder_only"
        elif len(set(sets)) == 1:
            category = "multiplicity_changed_same_set"
        else:
            category = "tool_set_changed"
        for scope in scopes:
            buckets[scope]["sequence_varying"] += 1
            buckets[scope][category] += 1

    rows = []
    scope_order = ["__all__", "__tool_calling__", *MODEL_ORDER]
    for scope in scope_order:
        counts = buckets.get(scope)
        if not counts:
            continue
        unanimous = counts["unanimous_groups"]
        varying = counts["sequence_varying"]
        scope_names = {
            "__all__": "All configurations",
            "__tool_calling__": "Tool-calling configurations",
        }
        rows.append(
            {
                "scope": scope,
                "scope_name": scope_names[scope]
                if scope in scope_names
                else MODEL_NAMES[scope],
                "unanimous_groups": unanimous,
                "sequence_varying": varying,
                "reorder_only": counts["reorder_only"],
                "multiplicity_changed_same_set": counts["multiplicity_changed_same_set"],
                "tool_set_changed": counts["tool_set_changed"],
                "sequence_variation_rate": varying / unanimous,
                "tool_set_change_rate": counts["tool_set_changed"] / unanimous,
            }
        )
    return pd.DataFrame(rows)


CONTROL_OR_ACTION_TOOLS = {
    "compliance": {"check_sanctions"},
    "dataops": {"apply_fix", "escalate_to_human", "validate_fix"},
}

EVIDENCE_CONTACT_TOOLS = {
    "compliance": {"calculate_risk_score", "get_customer_profile"},
    "dataops": {
        "get_exception_details",
        "get_historical_fixes",
        "query_reference_data",
    },
}


def rq2_tool_set_changes(
    groups: dict[CaseKey, list[Episode]],
) -> pd.DataFrame:
    """Describe which tool contacts vary in unanimous-decision groups.

    The historical fixtures do not define one required reference path.  The
    artifact therefore reports presence variation, not "omission" or
    "addition": either directional label would require a case-specific
    normative path that the study does not have.
    """
    rows: list[dict[str, object]] = []
    for (task, model, case_id), episodes in sorted(groups.items()):
        if len({episode.decision for episode in episodes}) != 1:
            continue
        tool_sets = [set(episode.path) for episode in episodes]
        canonical_sets = {tuple(sorted(tool_set)) for tool_set in tool_sets}
        if len(canonical_sets) == 1:
            continue
        union = set().union(*tool_sets)
        intersection = set.intersection(*tool_sets)
        variable = union - intersection
        control_tools = sorted(variable & CONTROL_OR_ACTION_TOOLS[task])
        evidence_tools = sorted(variable & EVIDENCE_CONTACT_TOOLS[task])
        unknown = sorted(
            variable - CONTROL_OR_ACTION_TOOLS[task] - EVIDENCE_CONTACT_TOOLS[task]
        )
        if unknown:
            raise AssertionError(
                f"unclassified variable tools for {task}/{model}/{case_id}: {unknown}"
            )
        rows.append(
            {
                "task": task,
                "model": model,
                "model_name": MODEL_NAMES[model],
                "case_id": case_id,
                "n_runs": len(episodes),
                "variable_tool_names": json.dumps(sorted(variable)),
                "variable_control_or_action_tools": json.dumps(control_tools),
                "variable_evidence_contact_tools": json.dumps(evidence_tools),
                "has_control_or_action_variation": bool(control_tools),
                "has_evidence_contact_variation": bool(evidence_tools),
                "directionality": "presence_varies_no_normative_reference_path",
            }
        )
    return pd.DataFrame(rows)


def zero_tool_inclusion_sensitivity(
    all_groups: dict[CaseKey, list[Episode]],
    primary_groups: dict[CaseKey, list[Episode]],
) -> pd.DataFrame:
    """Show the pooled RQ2 denominator with and without zero-tool stacks."""
    rows: list[dict[str, object]] = []
    for scope, groups in (
        (
            "primary_configurations_with_observed_tool_use",
            primary_groups,
        ),
        ("all_including_zero_tool_configurations", all_groups),
    ):
        decomposition = rq2_decomposition(groups, case_frame(groups))
        pooled = decomposition[decomposition["scope"] == "__all__"].iloc[0]
        rows.append(
            {
                "scope": scope,
                "unanimous_groups": int(pooled["unanimous_groups"]),
                "sequence_varying": int(pooled["sequence_varying"]),
                "sequence_variation_rate": float(pooled["sequence_variation_rate"]),
                "interpretation": (
                    "descriptive pooled sensitivity; configuration-specific rows unchanged"
                ),
            }
        )
    return pd.DataFrame(rows)


def flash_leave_one_case_out(case_df: pd.DataFrame) -> pd.DataFrame:
    """Check whether Flash task gaps are concentrated in individual cases."""
    rows: list[dict[str, object]] = []
    flash = case_df[case_df["model"] == "gemini-2.0-flash"]
    for task, task_df in flash.groupby("task", sort=True):
        full_gap = float(task_df["gap"].mean())
        n_cases = len(task_df)
        for _, case in task_df.sort_values("case_id").iterrows():
            leave_one_out_gap = (float(task_df["gap"].sum()) - float(case["gap"])) / (
                n_cases - 1
            )
            rows.append(
                {
                    "model": "gemini-2.0-flash",
                    "model_name": MODEL_NAMES["gemini-2.0-flash"],
                    "task": task,
                    "left_out_case_id": case["case_id"],
                    "n_cases_full": n_cases,
                    "n_cases_leave_one_out": n_cases - 1,
                    "full_gap": full_gap,
                    "leave_one_out_gap": leave_one_out_gap,
                    "absolute_change": abs(leave_one_out_gap - full_gap),
                }
            )
    return pd.DataFrame(rows)


def illustrative_shadow_flag_load(case_df: pd.DataFrame) -> pd.DataFrame:
    """Count one illustrative trigger: unanimous decision, varying sequence."""
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        model_df = case_df[case_df["model"] == model]
        flagged = model_df[(model_df["dar"] == 1.0) & (model_df["tar"] < 1.0)]
        rows.append(
            {
                "model": model,
                "model_name": MODEL_NAMES[model],
                "observed_groups": len(model_df),
                "flagged_groups": len(flagged),
                "flags_per_100_observed_groups": 100 * len(flagged) / len(model_df),
                "illustrative_trigger": "dar_equals_1_and_tar_sequence_below_1",
                "materiality_assessed": False,
            }
        )
    return pd.DataFrame(rows)


def case_cluster_bootstrap(case_df: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap cases within tasks, conditional on the observed replays.

    These are the intervals reported in the draft.  They quantify variation
    across the fixed case population represented by the sample; they do not
    absorb the finite-repeat uncertainty inside a case.  No finite-repeat
    interval is claimed: with only three API repeats, naively resampling a
    nonsmooth modal fraction is strongly shifted.  Independently logged
    replays are the remedy.
    """
    rows = []
    for model in MODEL_ORDER:
        model_df = case_df[case_df["model"] == model]
        if model_df.empty:
            continue
        per_task = {
            task: group[list(METRIC_COLUMNS)].to_numpy(dtype=float)
            for task, group in model_df.groupby("task", sort=True)
        }
        point = np.mean([values.mean(axis=0) for values in per_task.values()], axis=0)
        rng = np.random.default_rng(SEED)
        draws = np.zeros((N_BOOTSTRAP, len(METRIC_COLUMNS)), dtype=float)
        for values in per_task.values():
            indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
            draws += values[indices].mean(axis=1)
        draws /= len(per_task)
        for metric_index, metric in enumerate(METRIC_COLUMNS):
            low, high = np.quantile(draws[:, metric_index], [0.025, 0.975])
            rows.append(
                {
                    "model": model,
                    "model_name": MODEL_NAMES[model],
                    "metric": metric,
                    "point": point[metric_index],
                    "ci_low": low,
                    "ci_high": high,
                    "n_bootstrap": N_BOOTSTRAP,
                    "conditioning": "conditional_on_observed_replays",
                }
            )
    return pd.DataFrame(rows)


LOCAL_TOOL_MODELS = MODEL_ORDER[:4]
CONTROLLED_API_MODELS = ("gemini-2.0-flash", "gemini-2.5-pro")
N_SUBSAMPLES = 500
SUBSAMPLE_REPLAYS = 3
N_PERMUTATIONS = 10_000
FALLBACK_RATES = (0.01, 0.02, 0.05, 0.10)


def _aggregate_from_case_values(per_task_values: dict[str, np.ndarray]) -> np.ndarray:
    """Equal-weight task mean of per-case metric vectors (paper estimator)."""
    return np.mean([values.mean(axis=0) for values in per_task_values.values()], axis=0)


def n3_subsample_sensitivity(groups: dict[CaseKey, list[Episode]]) -> pd.DataFrame:
    """Subsample the 8-replay local configurations to 3 replays per case.

    For each of ``N_SUBSAMPLES`` draws, sample 3 of 8 episodes per case without
    replacement, recompute per-case DAR/TARseq/gap, and aggregate exactly as the
    paper does (task means, equal task weights).  Reported medians and 2.5/97.5
    percentiles describe subsampling variability, conditional on the observed
    eight replays; they are not population intervals.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for model in LOCAL_TOOL_MODELS:
        model_groups = {key: episodes for key, episodes in groups.items() if key[1] == model}
        for episodes in model_groups.values():
            assert len(episodes) == 8, "local groups must have exactly 8 replays"
        draws = np.zeros((N_SUBSAMPLES, 3), dtype=float)  # dar, tar, gap
        for draw_index in range(N_SUBSAMPLES):
            per_task: dict[str, list[list[float]]] = defaultdict(list)
            for (task, _model, _case_id), episodes in model_groups.items():
                chosen = rng.choice(len(episodes), size=SUBSAMPLE_REPLAYS, replace=False)
                sample = [episodes[i] for i in chosen]
                decisions = [episode.decision for episode in sample]
                paths = [episode.path for episode in sample]
                dar = agreement(decisions)
                tar = agreement(paths)
                per_task[task].append([dar, tar, dar - tar])
            draws[draw_index] = _aggregate_from_case_values(
                {task: np.array(values) for task, values in per_task.items()}
            )
        for metric_index, metric in enumerate(("dar", "tar", "gap")):
            low, median, high = np.quantile(draws[:, metric_index], [0.025, 0.5, 0.975])
            rows.append(
                {
                    "model": model,
                    "model_name": MODEL_NAMES[model],
                    "metric": metric,
                    "median": median,
                    "p2_5": low,
                    "p97_5": high,
                    "n_subsamples": N_SUBSAMPLES,
                    "subsample_replays": SUBSAMPLE_REPLAYS,
                    "conditioning": "conditional_on_observed_8_replays",
                }
            )
    return pd.DataFrame(rows)


def _aggregate_case_state_dar(
    case_states: list[dict[str, object]],
) -> float:
    per_task: dict[str, list[float]] = defaultdict(list)
    for state in case_states:
        task = str(state["task"])
        per_task[task].append(int(state["modal"]) / int(state["n"]))
    return float(np.mean([np.mean(values) for values in per_task.values()]))


def _fallback_marginal(
    state: dict[str, object],
    cases_per_task: dict[str, int],
    n_tasks: int,
) -> float:
    modal = int(state["modal"])
    if modal <= 1:
        return 0.0
    task = str(state["task"])
    return (1.0 / int(state["n"])) / cases_per_task[task] / n_tasks


def parser_fallback_bound(groups: dict[CaseKey, list[Episode]]) -> pd.DataFrame:
    """Adversarial worst-case bound on silent parser-fallback contamination.

    For each controlled API configuration and fallback rate k, assume
    ceil(k * episodes) episodes carried a silently substituted label and flip
    them adversarially: each flip removes one episode from its case's modal
    decision class (greedy over the largest marginal reduction in the
    task-weighted aggregate DAR).  TAR is unchanged because the fallback
    affected only the decision channel.  Reports the resulting DAR and paired
    gap, plus the exact crossing rate at which the gap reaches zero.
    """
    rows = []
    for model in CONTROLLED_API_MODELS:
        model_groups = {key: episodes for key, episodes in groups.items() if key[1] == model}
        n_episodes = sum(len(episodes) for episodes in model_groups.values())
        tasks = sorted({key[0] for key in model_groups})
        cases_per_task = {
            task: sum(1 for key in model_groups if key[0] == task) for task in tasks
        }
        # Per-case state: current modal count and replay count.
        case_states = []
        per_task_tar: dict[str, list[float]] = defaultdict(list)
        for (task, _m, _c), episodes in model_groups.items():
            decisions = [episode.decision for episode in episodes]
            paths = [episode.path for episode in episodes]
            modal_count = Counter(decisions).most_common(1)[0][1]
            case_states.append({"task": task, "n": len(episodes), "modal": modal_count})
            per_task_tar[task].append(agreement(paths))
        tar_aggregate = float(np.mean([np.mean(values) for values in per_task_tar.values()]))

        baseline_gap = _aggregate_case_state_dar(case_states) - tar_aggregate
        flips_applied = 0
        crossing_flips = None
        results_at = {}
        max_flips = int(np.ceil(max(FALLBACK_RATES) * n_episodes))
        targets = sorted(
            {rate: int(np.ceil(rate * n_episodes)) for rate in FALLBACK_RATES}.items()
        )
        target_index = 0
        while flips_applied <= max_flips:
            current_dar = _aggregate_case_state_dar(case_states)
            gap_now = current_dar - tar_aggregate
            if crossing_flips is None and gap_now <= 0:
                crossing_flips = flips_applied
            while target_index < len(targets) and flips_applied == targets[target_index][1]:
                rate = targets[target_index][0]
                results_at[rate] = (current_dar, gap_now)
                target_index += 1
            if flips_applied == max_flips:
                break
            best = max(
                case_states,
                key=lambda state: _fallback_marginal(state, cases_per_task, len(tasks)),
            )
            if _fallback_marginal(best, cases_per_task, len(tasks)) == 0.0:
                break
            best["modal"] = int(best["modal"]) - 1
            flips_applied += 1
        # Continue past the grid to find the crossing if not yet reached.
        while crossing_flips is None:
            best = max(
                case_states,
                key=lambda state: _fallback_marginal(state, cases_per_task, len(tasks)),
            )
            if _fallback_marginal(best, cases_per_task, len(tasks)) == 0.0:
                break
            best["modal"] = int(best["modal"]) - 1
            flips_applied += 1
            if _aggregate_case_state_dar(case_states) - tar_aggregate <= 0:
                crossing_flips = flips_applied
        for rate, (dar_value, gap_value) in sorted(results_at.items()):
            rows.append(
                {
                    "model": model,
                    "model_name": MODEL_NAMES[model],
                    "fallback_rate": rate,
                    "n_fallback_episodes": int(np.ceil(rate * n_episodes)),
                    "n_episodes": n_episodes,
                    "adversarial_dar": dar_value,
                    "adversarial_gap": gap_value,
                    "baseline_gap": baseline_gap,
                    "gap_sign_flipped": gap_value <= 0,
                }
            )
        rows.append(
            {
                "model": model,
                "model_name": MODEL_NAMES[model],
                "fallback_rate": (
                    crossing_flips / n_episodes if crossing_flips is not None else float("nan")
                ),
                "n_fallback_episodes": crossing_flips,
                "n_episodes": n_episodes,
                "adversarial_dar": float("nan"),
                "adversarial_gap": 0.0,
                "baseline_gap": baseline_gap,
                "gap_sign_flipped": True,
            }
        )
    return pd.DataFrame(rows)


def first_replay_determinism(groups: dict[CaseKey, list[Episode]]) -> pd.DataFrame:
    """Lineage comparison: first-replay anchoring versus modal agreement.

    The earlier DFAH harness anchored Action/Decision Determinism to the first
    replay: a replay agreed only if it matched replay one.  Here we recompute
    that estimand on the same retained name-only slice (fraction of all
    replays, including the anchor, equal to replay one's value) and report the
    task-weighted aggregate next to the modal DAR/TARseq used in this paper.
    Modal agreement is by construction at least first-anchored agreement.
    Signature Determinism cannot be recomputed: arguments and results were not
    retained in the compact logs.
    """
    rows = []
    models = sorted({key[1] for key in groups})
    for model in models:
        per_task: dict[str, list[list[float]]] = defaultdict(list)
        for (task, group_model, _case_id), episodes in groups.items():
            if group_model != model:
                continue
            ordered = sorted(episodes, key=lambda episode: episode.run)
            first = ordered[0]
            decisions = [episode.decision for episode in ordered]
            paths = [episode.path for episode in ordered]
            first_dar = sum(decision == first.decision for decision in decisions) / len(ordered)
            first_tar = sum(path == first.path for path in paths) / len(ordered)
            per_task[task].append(
                [
                    agreement(decisions),
                    agreement(paths),
                    first_dar,
                    first_tar,
                ]
            )
        aggregate = _aggregate_from_case_values(
            {task: np.array(values) for task, values in per_task.items()}
        )
        rows.append(
            {
                "model": model,
                "model_name": MODEL_NAMES[model],
                "modal_dar": aggregate[0],
                "modal_tar": aggregate[1],
                "first_anchored_dar": aggregate[2],
                "first_anchored_tar": aggregate[3],
                "dar_delta": aggregate[0] - aggregate[2],
                "tar_delta": aggregate[1] - aggregate[3],
                "anchor": "first_replay_inclusive",
            }
        )
    return pd.DataFrame(rows)


def paired_permutation_tests(case_df: pd.DataFrame) -> pd.DataFrame:
    """Within-task sign-flip permutation test on per-case paired gaps.

    Null hypothesis: the per-case paired gap is sign-symmetric around zero
    (exchangeability of the DAR/TAR labels within a case).  The statistic is
    the task-weighted mean gap; signs are flipped independently per case within
    each task.  This tests exchangeability of the paired gap in the observed
    corpus, not a population effect.
    """
    rows = []
    for model in CONTROLLED_API_MODELS:
        model_df = case_df[case_df["model"] == model]
        per_task = {
            task: group["gap"].to_numpy(dtype=float)
            for task, group in model_df.groupby("task", sort=True)
        }
        observed = float(np.mean([values.mean() for values in per_task.values()]))
        rng = np.random.default_rng(SEED)
        permuted = np.zeros(N_PERMUTATIONS, dtype=float)
        for task_values in per_task.values():
            signs = rng.choice((-1.0, 1.0), size=(N_PERMUTATIONS, len(task_values)))
            permuted += (signs * task_values).mean(axis=1)
        permuted /= len(per_task)
        exceed = int(np.sum(np.abs(permuted) >= abs(observed) - 1e-12))
        p_value = (1 + exceed) / (1 + N_PERMUTATIONS)
        rows.append(
            {
                "model": model,
                "model_name": MODEL_NAMES[model],
                "observed_gap": observed,
                "p_value": p_value,
                "n_permutations": N_PERMUTATIONS,
                "seed": SEED,
                "null": "within_task_sign_symmetry_of_paired_gap",
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture_rows = load_fixture(args.fixture)
    lineage = corpus_lineage(fixture_rows)
    all_groups = load_groups(fixture_rows)
    channel_eligibility = trajectory_channel_eligibility(fixture_rows)
    exclusion_rows = []
    for model, reason in ZERO_TOOL_CONFIGURATION_EXCLUSIONS.items():
        model_groups = {
            key: episodes for key, episodes in all_groups.items() if key[1] == model
        }
        episode_count = sum(len(episodes) for episodes in model_groups.values())
        observed_tool_calls = sum(
            len(episode.path) for episodes in model_groups.values() for episode in episodes
        )
        if not model_groups or observed_tool_calls != 0:
            raise AssertionError(
                f"non-agentic exclusion changed for {model}: "
                f"groups={len(model_groups)}, tool_calls={observed_tool_calls}"
            )
        exclusion_rows.append(
            {
                "model": model,
                "tasks": len({key[0] for key in model_groups}),
                "case_groups": len(model_groups),
                "episodes": episode_count,
                "observed_tool_calls": observed_tool_calls,
                "exclusion_reason": reason,
                "capability_inference": "none; exact configuration/harness observation only",
            }
        )
    exclusions = pd.DataFrame(exclusion_rows)
    groups = {
        key: episodes
        for key, episodes in all_groups.items()
        if key[1] not in ZERO_TOOL_CONFIGURATION_EXCLUSIONS
    }
    cases = case_frame(groups)
    tasks = task_summary(cases)
    summary = task_weighted_summary(cases).merge(
        tasks.groupby("model", as_index=False).agg(
            decision_concentration=("decision_concentration", "mean")
        ),
        on="model",
        validate="one_to_one",
    )
    rq2 = rq2_decomposition(groups, cases)
    rq2_tool_sets = rq2_tool_set_changes(groups)
    zero_tool_sensitivity = zero_tool_inclusion_sensitivity(all_groups, groups)
    flash_loo = flash_leave_one_case_out(cases)
    flag_load = illustrative_shadow_flag_load(cases)
    cluster_cis = case_cluster_bootstrap(cases)
    n3_subsample = n3_subsample_sensitivity(groups)
    fallback_bound = parser_fallback_bound(groups)
    permutation = paired_permutation_tests(cases)
    first_replay = first_replay_determinism(groups)

    cases.to_csv(args.output_dir / "analysis_case_level.csv", index=False)
    summary.to_csv(args.output_dir / "analysis_summary.csv", index=False)
    tasks.to_csv(args.output_dir / "analysis_task_level.csv", index=False)
    rq2.to_csv(args.output_dir / "analysis_rq2_decomposition.csv", index=False)
    rq2_tool_sets.to_csv(args.output_dir / "analysis_rq2_tool_set_changes.csv", index=False)
    zero_tool_sensitivity.to_csv(
        args.output_dir / "analysis_zero_tool_inclusion_sensitivity.csv",
        index=False,
    )
    flash_loo.to_csv(args.output_dir / "analysis_flash_leave_one_case_out.csv", index=False)
    flag_load.to_csv(args.output_dir / "analysis_shadow_flag_load.csv", index=False)
    cluster_cis.to_csv(args.output_dir / "analysis_cluster_cis.csv", index=False)
    n3_subsample.to_csv(args.output_dir / "analysis_n3_subsample.csv", index=False)
    fallback_bound.to_csv(args.output_dir / "analysis_fallback_bound.csv", index=False)
    permutation.to_csv(args.output_dir / "analysis_permutation.csv", index=False)
    first_replay.to_csv(args.output_dir / "analysis_first_replay.csv", index=False)
    exclusions.to_csv(
        args.output_dir / "analysis_zero_tool_configuration_exclusions.csv",
        index=False,
    )
    channel_eligibility.to_csv(
        args.output_dir / "analysis_channel_eligibility.csv", index=False
    )
    lineage.to_csv(args.output_dir / "analysis_corpus_lineage.csv", index=False)

    print(
        f"Corrected v2 slice: {len(cases)} repeated case groups, "
        f"{int(cases['n_runs'].sum())} episodes, tasks={','.join(EVAL_TASKS)}"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
