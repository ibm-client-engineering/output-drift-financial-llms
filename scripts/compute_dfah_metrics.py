#!/usr/bin/env python3
"""Compute DFAH-Bench metrics on the full replay corpus.

Current corpus status (2026-04-06):
  - Raw checked-in corpus: 8,129 episodes across 1,340 case groups
  - Current analyzed subset: 8,127 episodes across 1,338 case groups
  - Two raw single-run groups are excluded from case-level metrics
    because DAR/TAR/ECD require repeated runs on the same case

Evaluates the kill criterion for the NeurIPS 2026 E&D submission and
produces the working CSVs used for tables/figures.

Metrics computed per case group (case_id x benchmark x model):
  - DAR: Decision Agreement Rate (fraction agreeing with modal decision)
  - TAR: Trajectory Agreement Rate (fraction matching modal tool sequence)
  - DAR-TAR gap: the central finding — decision stability exceeding trajectory stability
  - within-case DCB: replay-level decision concentration within a case group
    (the paper's Table 1 cross-case DCB is computed separately by
    scripts/compute_dcb_across_case.py)
  - ECD: Evidence Contact Divergence (mean pairwise Jaccard distance)
  - SCDE: DAR * ECD (same conclusion, different evidence)

Usage:
    python scripts/compute_dfah_metrics.py
    python scripts/compute_dfah_metrics.py --run-logs-dir path/to/run_logs
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure bench/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.metrics.dcb import compute_dcb
from bench.metrics.ecd import compute_ecd
from bench.spec.schema import (
    DivergenceChannel,
    ReplayEpisode,
    available_channels,
    load_episodes,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_run_logs_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "econometrics" / "benchmarks" / "results" / "run_logs"


def load_all_episodes(run_logs_dir: Path) -> List[ReplayEpisode]:
    """Walk benchmark/model directories and load all episodes."""
    all_episodes: List[ReplayEpisode] = []
    for benchmark_dir in sorted(run_logs_dir.iterdir()):
        if not benchmark_dir.is_dir():
            continue
        for model_dir in sorted(benchmark_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            eps = load_episodes(model_dir)
            all_episodes.extend(eps)
    return all_episodes


# ---------------------------------------------------------------------------
# Case-level metric computation
# ---------------------------------------------------------------------------

CaseKey = Tuple[str, str, str]  # (case_id, benchmark, model)


def group_by_case(
    episodes: List[ReplayEpisode],
) -> Dict[CaseKey, List[ReplayEpisode]]:
    groups: Dict[CaseKey, List[ReplayEpisode]] = defaultdict(list)
    for ep in episodes:
        key = (ep.case_id, ep.benchmark, ep.metadata.model_name or "unknown")
        groups[key].append(ep)
    return dict(groups)


def _compute_tar(
    episodes: List[ReplayEpisode],
) -> Tuple[Optional[float], int, Optional[str]]:
    """Trajectory Agreement Rate via exact sequence match.

    Denominator convention: TAR is computed over the runs that made at
    least one tool call. Runs with zero tool calls are treated as having no
    trajectory channel (not as an empty sequence), so they do not enter the
    numerator or denominator. When NO run in the group made tool calls the
    channel is absent and (None, 0, None) is returned — the group is then
    reported as missing trajectory data rather than as TAR = 1.0.
    This is the convention used for all published TAR numbers; counting
    zero-tool runs as empty sequences instead would lower TAR for mixed
    groups (and widen the DAR-TAR gap), so the published convention is the
    conservative one for the paper's central claim.

    Returns (tar, n_unique_trajectories, modal_trajectory_str).
    Returns (None, 0, None) when no episodes have tool calls.
    """
    sequences: List[Tuple[str, ...]] = []
    for ep in episodes:
        if ep.tool_calls:
            seq = tuple(tc.name for tc in ep.tool_calls)
            sequences.append(seq)

    if not sequences:
        return None, 0, None

    counts = Counter(sequences)
    modal_seq, modal_count = counts.most_common(1)[0]
    tar = modal_count / len(sequences)
    modal_str = " -> ".join(modal_seq)
    return tar, len(counts), modal_str


def compute_case_metrics(
    key: CaseKey,
    episodes: List[ReplayEpisode],
) -> dict:
    case_id, benchmark, model = key
    n_runs = len(episodes)

    # --- Decisions ---
    # Denominator convention: DAR is computed over runs that produced a
    # non-empty decision label. Episodes with an empty decision are treated
    # as missing the decision channel rather than as an "abstain" category.
    decisions = [ep.decision.label for ep in episodes if ep.decision.label]

    dar: Optional[float] = None
    modal_decision: Optional[str] = None
    n_unique_decisions = 0
    if decisions:
        counts = Counter(decisions)
        modal_decision, modal_count = counts.most_common(1)[0]
        dar = modal_count / len(decisions)
        n_unique_decisions = len(counts)

    # --- DCB ---
    dcb: Optional[float] = None
    if decisions:
        try:
            dcb = compute_dcb(decisions, benchmark=benchmark).dcb
        except (ValueError, KeyError):
            try:
                dcb = compute_dcb(decisions).dcb
            except ValueError:
                pass

    # --- Trajectory ---
    has_trajectory = any(
        DivergenceChannel.TRAJECTORY in available_channels(ep) for ep in episodes
    )
    tar, n_unique_trajectories, modal_trajectory = _compute_tar(episodes)
    dar_tar_gap = (dar - tar) if (dar is not None and tar is not None) else None

    # --- Evidence contacts ---
    has_evidence = any(
        DivergenceChannel.EVIDENCE_CONTACT in available_channels(ep)
        for ep in episodes
    )
    ecd: Optional[float] = None
    scde: Optional[float] = None
    if has_evidence:
        evidence_sets: List[set] = []
        evidence_decisions: List[str] = []
        for ep in episodes:
            if ep.evidence_contacts:
                ev_set = {ec.source_id for ec in ep.evidence_contacts}
                evidence_sets.append(ev_set)
                evidence_decisions.append(ep.decision.label)

        if len(evidence_sets) >= 2:
            try:
                ecd_result = compute_ecd(evidence_sets, decisions=evidence_decisions)
                ecd = ecd_result.ecd
                scde = ecd_result.scde
            except ValueError:
                pass

    return {
        "case_id": case_id,
        "benchmark": benchmark,
        "model": model,
        "n_runs": n_runs,
        "dar": dar,
        "modal_decision": modal_decision,
        "n_unique_decisions": n_unique_decisions,
        "dcb": dcb,
        "has_trajectory": has_trajectory,
        "tar": tar,
        "n_unique_trajectories": n_unique_trajectories,
        "dar_tar_gap": dar_tar_gap,
        "has_evidence": has_evidence,
        "ecd": ecd,
        "scde": scde,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_task_level(case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (benchmark, model), grp in case_df.groupby(["benchmark", "model"]):
        traj = grp[grp["has_trajectory"] & grp["tar"].notna()]
        ev = grp[grp["has_evidence"] & grp["ecd"].notna()]

        rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "n_cases": len(grp),
                "n_episodes": int(grp["n_runs"].sum()),
                "mean_dar": grp["dar"].mean(),
                "std_dar": grp["dar"].std(),
                "mean_dcb": grp["dcb"].mean(),
                "traj_coverage": f"{len(traj)}/{len(grp)}",
                "mean_tar": traj["tar"].mean() if len(traj) else None,
                "std_tar": traj["tar"].std() if len(traj) else None,
                "mean_dar_tar_gap": traj["dar_tar_gap"].mean()
                if len(traj)
                else None,
                "ev_coverage": f"{len(ev)}/{len(grp)}",
                "mean_ecd": ev["ecd"].mean() if len(ev) else None,
                "mean_scde": ev["scde"].mean() if len(ev) else None,
            }
        )
    return pd.DataFrame(rows)


def aggregate_model_level(task_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, grp in task_df.groupby("model"):
        rows.append(
            {
                "model": model,
                "n_benchmarks": len(grp),
                "total_episodes": int(grp["n_episodes"].sum()),
                "mean_dar": grp["mean_dar"].mean(),
                "mean_dcb": grp["mean_dcb"].mean(),
                "mean_tar": grp["mean_tar"].dropna().mean()
                if grp["mean_tar"].notna().any()
                else None,
                "mean_dar_tar_gap": grp["mean_dar_tar_gap"].dropna().mean()
                if grp["mean_dar_tar_gap"].notna().any()
                else None,
                "mean_ecd": grp["mean_ecd"].dropna().mean()
                if grp["mean_ecd"].notna().any()
                else None,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Kill criterion
# ---------------------------------------------------------------------------

def evaluate_kill_criterion(
    case_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Among high-DAR cases with trajectory data, how many show divergence?

    Returns (overall_df, by_model_df).
    """

    eligible = case_df[
        (case_df["dar"] >= 0.9)
        & case_df["has_trajectory"]
        & case_df["tar"].notna()
    ].copy()

    n = len(eligible)

    print(f"\n{'=' * 72}")
    print("KILL CRITERION EVALUATION")
    print(f"{'=' * 72}")

    if n == 0:
        print("No eligible cases (DAR >= 0.9 with trajectory data).")
        return pd.DataFrame(), pd.DataFrame()

    # Three divergence thresholds
    any_var = eligible[eligible["n_unique_trajectories"] > 1]
    moderate = eligible[eligible["tar"] < 0.9]
    strong = eligible[eligible["tar"] < 0.7]

    pct_any = len(any_var) / n * 100
    pct_mod = len(moderate) / n * 100
    pct_str = len(strong) / n * 100

    print(f"Eligible cases (DAR >= 0.9, trajectory available): {n}")
    print()
    print(
        f"  Any traj variation  (n_unique > 1):  "
        f"{len(any_var):>4} / {n} = {pct_any:5.1f}%"
    )
    print(
        f"  Moderate divergence (TAR < 0.9):      "
        f"{len(moderate):>4} / {n} = {pct_mod:5.1f}%"
    )
    print(
        f"  Strong divergence   (TAR < 0.7):      "
        f"{len(strong):>4} / {n} = {pct_str:5.1f}%"
    )
    print()

    # Verdict
    if pct_any < 10:
        print(
            "  FAIL: <10% of high-DAR cases show ANY trajectory variation.\n"
            "  The central claim is not supported. Paper viability risk: HIGH."
        )
    elif pct_mod < 10:
        print(
            "  WEAK: Some variation exists but <10% show moderate divergence.\n"
            "  Signal may be too weak for main-track submission."
        )
    else:
        print(
            f"  PASS: {pct_mod:.1f}% of high-DAR cases show moderate trajectory"
            f" divergence.\n"
            "  The central claim is supported. Proceed with paper writing."
        )

    print(f"{'=' * 72}")

    # Per-model breakdown within eligible set — persisted to CSV so the
    # per-model diverger rates quoted in the paper (e.g. 55.6% Claude
    # Sonnet, 56.6% Gemini 2.5 Pro) are regenerable artifacts, not
    # stdout-only numbers.
    print("\nPer-model breakdown (eligible cases only):")
    by_model_rows = []
    for model, mg in eligible.groupby("model"):
        mn = len(mg)
        m_any = int((mg["n_unique_trajectories"] > 1).sum())
        m_mod = int((mg["tar"] < 0.9).sum())
        m_str = int((mg["tar"] < 0.7).sum())
        print(
            f"  {model:<35s}  n={mn:>3}  "
            f"any_var={m_any:>3} ({m_any/mn*100:5.1f}%)  "
            f"moderate={m_mod:>3} ({m_mod/mn*100:5.1f}%)"
        )
        by_model_rows.append(
            {"model": model, "n_eligible": mn,
             "n_any_variation": m_any, "pct_any_variation": m_any / mn * 100,
             "n_moderate_tar_lt_0.9": m_mod, "pct_moderate": m_mod / mn * 100,
             "n_strong_tar_lt_0.7": m_str, "pct_strong": m_str / mn * 100}
        )

    kill_df = pd.DataFrame(
        [
            {"criterion": "any_variation", "n_eligible": n,
             "n_divergent": len(any_var), "pct": pct_any},
            {"criterion": "moderate_tar_lt_0.9", "n_eligible": n,
             "n_divergent": len(moderate), "pct": pct_mod},
            {"criterion": "strong_tar_lt_0.7", "n_eligible": n,
             "n_divergent": len(strong), "pct": pct_str},
        ]
    )
    return kill_df, pd.DataFrame(by_model_rows)


# ---------------------------------------------------------------------------
# Headline findings
# ---------------------------------------------------------------------------

def print_headline_findings(case_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    """Print the key numbers for the paper abstract."""
    print(f"\n{'=' * 72}")
    print("HEADLINE FINDINGS")
    print(f"{'=' * 72}")

    traj = case_df[case_df["has_trajectory"] & case_df["tar"].notna()]
    high_dar_traj = traj[traj["dar"] >= 0.9]

    if len(high_dar_traj) > 0:
        mean_gap = high_dar_traj["dar_tar_gap"].mean()
        median_gap = high_dar_traj["dar_tar_gap"].median()
        print(
            f"\nAmong {len(high_dar_traj)} cases with DAR >= 0.9 and trajectory"
            f" data:"
        )
        print(f"  Mean  DAR-TAR gap: {mean_gap:.3f}")
        print(f"  Median DAR-TAR gap: {median_gap:.3f}")
        print(f"  Mean TAR:          {high_dar_traj['tar'].mean():.3f}")
        print(f"  Max DAR-TAR gap:   {high_dar_traj['dar_tar_gap'].max():.3f}")

    # Models with largest gap
    gap_models = model_df[model_df["mean_dar_tar_gap"].notna()].sort_values(
        "mean_dar_tar_gap", ascending=False
    )
    if len(gap_models) > 0:
        print("\nModels ranked by mean DAR-TAR gap (decision stability > trajectory):")
        for _, row in gap_models.iterrows():
            gap_str = f"{row['mean_dar_tar_gap']:.3f}" if pd.notna(row['mean_dar_tar_gap']) else "N/A"
            tar_str = f"{row['mean_tar']:.3f}" if pd.notna(row['mean_tar']) else "N/A"
            print(
                f"  {row['model']:<35s}  DAR={row['mean_dar']:.3f}  "
                f"TAR={tar_str}  gap={gap_str}"
            )

    # Legacy within-case DCB extremes. Table 1's cross-case DCB is computed
    # by scripts/compute_dcb_across_case.py.
    dcb_models = model_df.sort_values("mean_dcb", ascending=False)
    print("\nModels ranked by within-case DCB (legacy diagnostic, not Table 1 DCB):")
    for _, row in dcb_models.iterrows():
        print(f"  {row['model']:<35s}  DCB={row['mean_dcb']:.3f}")

    # ECD
    ecd_models = model_df[model_df["mean_ecd"].notna()].sort_values(
        "mean_ecd", ascending=False
    )
    if len(ecd_models) > 0:
        print("\nModels ranked by mean ECD (evidence divergence):")
        for _, row in ecd_models.iterrows():
            print(f"  {row['model']:<35s}  ECD={row['mean_ecd']:.3f}")

    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute DFAH-Bench metrics on the full replay corpus"
    )
    parser.add_argument(
        "--run-logs-dir", type=Path, default=None,
        help="Path to run_logs directory (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Directory for output CSVs (default: results/ next to bench/). "
            "Reproduction runs (e.g. `make reproduce-paper`) MUST pass a "
            "scratch directory here and diff against the committed reference "
            "CSVs — never regenerate the reference in place."
        ),
    )
    args = parser.parse_args()

    run_logs_dir = args.run_logs_dir or find_run_logs_dir()
    results_dir = args.output_dir or (Path(__file__).resolve().parent.parent / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print(f"Loading episodes from: {run_logs_dir}")
    episodes = load_all_episodes(run_logs_dir)
    print(f"Loaded {len(episodes)} episodes")

    groups = group_by_case(episodes)
    n_benchmarks = len({k[1] for k in groups})
    n_models = len({k[2] for k in groups})
    print(
        f"Found {len(groups)} case groups across "
        f"{n_benchmarks} benchmarks, {n_models} models"
    )

    skipped_rows = []
    for key, eps in sorted(groups.items()):
        if len(eps) < 2:
            case_id, benchmark, model = key
            skipped_rows.append(
                {
                    "case_id": case_id,
                    "benchmark": benchmark,
                    "model": model,
                    "n_runs": len(eps),
                }
            )

    if skipped_rows:
        skipped_episodes = sum(row["n_runs"] for row in skipped_rows)
        print(
            f"Excluding {len(skipped_rows)} single-run case groups "
            f"({skipped_episodes} episodes) from case-level metrics:"
        )
        for row in skipped_rows:
            print(
                f"  {row['benchmark']:<10s} {row['model']:<28s} "
                f"{row['case_id']} (n_runs={row['n_runs']})"
            )

    # Case-level metrics (require N >= 2)
    print("\nComputing case-level metrics...")
    case_rows = []
    for key, eps in sorted(groups.items()):
        if len(eps) < 2:
            continue
        case_rows.append(compute_case_metrics(key, eps))

    case_df = pd.DataFrame(case_rows)
    print(f"Computed metrics for {len(case_df)} case groups (N >= 2)")

    # Aggregations
    task_df = aggregate_task_level(case_df)
    model_df = aggregate_model_level(task_df)

    # Kill criterion
    kill_df, kill_by_model_df = evaluate_kill_criterion(case_df)

    # Headline findings
    print_headline_findings(case_df, model_df)

    # Print tables
    print(f"\n{'=' * 72}")
    print("TASK-LEVEL TABLE")
    print(f"{'=' * 72}")
    with pd.option_context("display.max_columns", 20, "display.width", 200):
        print(task_df.to_string(index=False, float_format="%.3f"))

    print(f"\n{'=' * 72}")
    print("MODEL-LEVEL TABLE")
    print(f"{'=' * 72}")
    with pd.option_context("display.max_columns", 20, "display.width", 200):
        print(model_df.to_string(index=False, float_format="%.3f"))

    # Write CSVs
    case_df.to_csv(results_dir / "dfah_case_level.csv", index=False)
    task_df.to_csv(results_dir / "dfah_task_level.csv", index=False)
    model_df.to_csv(results_dir / "dfah_model_level.csv", index=False)
    if len(kill_df) > 0:
        kill_df.to_csv(results_dir / "dfah_kill_criterion.csv", index=False)
    if len(kill_by_model_df) > 0:
        kill_by_model_df.to_csv(
            results_dir / "dfah_kill_criterion_by_model.csv", index=False
        )
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(
            results_dir / "dfah_skipped_case_groups.csv", index=False
        )

    print(f"\nCSVs written to {results_dir}/")
    for name in [
        "dfah_case_level.csv",
        "dfah_task_level.csv",
        "dfah_model_level.csv",
        "dfah_kill_criterion.csv",
        "dfah_kill_criterion_by_model.csv",
        "dfah_skipped_case_groups.csv",
    ]:
        p = results_dir / name
        if p.exists():
            n_rows = sum(1 for _ in open(p)) - 1
            print(f"  {name:<30s}  {n_rows} rows")


if __name__ == "__main__":
    main()
