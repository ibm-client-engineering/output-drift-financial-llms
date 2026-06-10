"""Lightweight aggregation helpers for building paper tables.

Aggregates replay episodes by case, task, and model. Outputs DataFrames
friendly for LaTeX table generation and plotting.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

from bench.metrics.dcb import compute_dcb
from bench.metrics.ecd import compute_ecd
from bench.spec.schema import (
    DivergenceChannel,
    ReplayEpisode,
    available_channels,
)


def _group_episodes(
    episodes: List[ReplayEpisode],
    key_fn,
) -> Dict[str, List[ReplayEpisode]]:
    """Group episodes by an arbitrary key function."""
    groups = defaultdict(list)
    for ep in episodes:
        groups[key_fn(ep)].append(ep)
    return dict(groups)


def aggregate_by_case(episodes: List[ReplayEpisode]) -> pd.DataFrame:
    """Aggregate metrics per case (same case_id across runs).

    Returns DataFrame with columns: case_id, benchmark, model, n_runs,
    dcb, n_unique_decisions, has_trajectory, has_evidence.
    """
    groups = _group_episodes(
        episodes, lambda ep: (ep.case_id, ep.benchmark, ep.metadata.model_name or "unknown")
    )

    rows = []
    for (case_id, benchmark, model), eps in sorted(groups.items()):
        decisions = [ep.decision.label for ep in eps if ep.decision.label]

        dcb_val = None
        if decisions:
            try:
                dcb_result = compute_dcb(decisions, benchmark=benchmark)
                dcb_val = dcb_result.dcb
            except (ValueError, KeyError):
                pass

        has_traj = any(
            DivergenceChannel.TRAJECTORY in available_channels(ep) for ep in eps
        )
        has_evidence = any(
            DivergenceChannel.EVIDENCE_CONTACT in available_channels(ep) for ep in eps
        )

        rows.append({
            "case_id": case_id,
            "benchmark": benchmark,
            "model": model,
            "n_runs": len(eps),
            "dcb": dcb_val,
            "n_unique_decisions": len(set(decisions)),
            "has_trajectory": has_traj,
            "has_evidence": has_evidence,
        })

    return pd.DataFrame(rows)


def aggregate_by_task(episodes: List[ReplayEpisode]) -> pd.DataFrame:
    """Aggregate metrics per benchmark task.

    Returns DataFrame with columns: benchmark, model, n_cases, n_episodes,
    mean_dcb, channel_coverage.
    """
    groups = _group_episodes(
        episodes, lambda ep: (ep.benchmark, ep.metadata.model_name or "unknown")
    )

    rows = []
    for (benchmark, model), eps in sorted(groups.items()):
        # Group by case for case-level DCB
        case_groups = _group_episodes(eps, lambda ep: ep.case_id)
        dcb_values = []
        for case_id, case_eps in case_groups.items():
            decisions = [ep.decision.label for ep in case_eps if ep.decision.label]
            if decisions:
                try:
                    dcb_values.append(compute_dcb(decisions, benchmark=benchmark).dcb)
                except (ValueError, KeyError):
                    pass

        # Channel coverage
        traj_count = sum(
            1 for ep in eps
            if DivergenceChannel.TRAJECTORY in available_channels(ep)
        )
        evidence_count = sum(
            1 for ep in eps
            if DivergenceChannel.EVIDENCE_CONTACT in available_channels(ep)
        )

        rows.append({
            "benchmark": benchmark,
            "model": model,
            "n_cases": len(case_groups),
            "n_episodes": len(eps),
            "mean_dcb": float(sum(dcb_values) / len(dcb_values)) if dcb_values else None,
            "trajectory_coverage": traj_count / len(eps) if eps else 0,
            "evidence_coverage": evidence_count / len(eps) if eps else 0,
        })

    return pd.DataFrame(rows)


def aggregate_by_model(episodes: List[ReplayEpisode]) -> pd.DataFrame:
    """Aggregate metrics per model across all benchmarks.

    Returns DataFrame with columns: model, n_benchmarks, n_episodes,
    mean_dcb, trajectory_coverage, evidence_coverage.
    """
    groups = _group_episodes(
        episodes, lambda ep: ep.metadata.model_name or "unknown"
    )

    rows = []
    for model, eps in sorted(groups.items()):
        benchmarks = set(ep.benchmark for ep in eps)

        all_decisions = [ep.decision.label for ep in eps if ep.decision.label]
        dcb_val = None
        if all_decisions:
            try:
                dcb_val = compute_dcb(all_decisions).dcb
            except ValueError:
                pass

        traj_count = sum(
            1 for ep in eps
            if DivergenceChannel.TRAJECTORY in available_channels(ep)
        )
        evidence_count = sum(
            1 for ep in eps
            if DivergenceChannel.EVIDENCE_CONTACT in available_channels(ep)
        )

        rows.append({
            "model": model,
            "n_benchmarks": len(benchmarks),
            "n_episodes": len(eps),
            "mean_dcb": dcb_val,
            "trajectory_coverage": traj_count / len(eps) if eps else 0,
            "evidence_coverage": evidence_count / len(eps) if eps else 0,
        })

    return pd.DataFrame(rows)
