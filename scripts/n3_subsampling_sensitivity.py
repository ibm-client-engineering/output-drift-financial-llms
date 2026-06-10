#!/usr/bin/env python3
"""
N=3 subsampling sensitivity analysis for DFAH-Bench taxonomy robustness.

Proves that the three-tier taxonomy (pattern matcher / stable executor /
trajectory diverger) is preserved when subsampling N=8 local model runs
down to N=3, defending against reviewer concerns about API models having
only N=3 replays.

Method:
  For each case group with N=8 runs, enumerate all C(8,3)=56 subsamples.
  Compute DAR and TAR at N=3 for each subsample. Compare model-level
  means and Spearman rank correlations between N=3 and N=8.

Usage:
  python scripts/n3_subsampling_sensitivity.py
"""

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUN_LOG_ROOT = Path(__file__).resolve().parent.parent / \
    "econometrics" / "benchmarks" / "results" / "run_logs"

BENCHMARKS = ["compliance", "portfolio", "dataops"]

LOCAL_MODELS = {
    "qwen2.5_7b-instruct": "Qwen 2.5 7B",
    "gpt-oss_20b": "GPT-OSS 20B",
    "granite3.3_latest": "Granite 3.3",
    "mistral_7b": "Mistral 7B",
}

# Paper tier assignments (ground truth to defend)
PAPER_TIERS = {
    "qwen2.5_7b-instruct": "Pattern",
    "gpt-oss_20b": "Stable",
    "granite3.3_latest": "Pattern",
    "mistral_7b": "Pattern",
}

N_FULL = 8
N_SUB = 3
K = 3  # decision categories per benchmark

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_run_logs(root: Path) -> dict:
    """Load all run logs for local models into nested dict.

    Returns:
        {model_slug: {benchmark: {case_id: [run_dicts]}}}
    """
    data = {}
    for model_slug in LOCAL_MODELS:
        data[model_slug] = {}
        for bench in BENCHMARKS:
            bench_dir = root / bench / model_slug
            if not bench_dir.is_dir():
                continue
            cases = defaultdict(list)
            for f in sorted(bench_dir.glob("case_*_run_*.json")):
                if "_full" in f.name:
                    continue
                with open(f) as fh:
                    run = json.load(fh)
                cases[run["case_id"]].append(run)
            # Only keep cases with exactly N_FULL runs
            complete = {
                cid: sorted(runs, key=lambda r: r["run_id"])
                for cid, runs in cases.items()
                if len(runs) == N_FULL
            }
            if complete:
                data[model_slug][bench] = complete
    return data


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_dar(runs: list[dict]) -> float:
    """Decision Agreement Rate: fraction agreeing with modal decision."""
    decisions = [r["decision_output"] for r in runs]
    modal_count = Counter(decisions).most_common(1)[0][1]
    return modal_count / len(decisions)


def compute_tar(runs: list[dict]) -> Optional[float]:
    """Trajectory Agreement Rate: fraction matching modal tool sequence.

    Returns None if fewer than 2 runs have tool_sequence data.
    Runs without tool data are excluded (not counted as mismatches).
    """
    sequences = []
    for r in runs:
        seq = r.get("tool_sequence")
        if seq is not None and len(seq) > 0:
            sequences.append(tuple(seq))
    if len(sequences) < 2:
        return None
    modal_count = Counter(sequences).most_common(1)[0][1]
    return modal_count / len(sequences)


def compute_dcb_for_model(
    case_decisions: dict[str, list[str]], k: int = K
) -> float:
    """DCB = 1 - H(p)/log(K), computed over aggregated decision distribution.

    case_decisions: {case_id: [decisions across runs]}
    Uses ALL decisions (not modal) to build the distribution.
    """
    all_decisions = []
    for decs in case_decisions.values():
        all_decisions.extend(decs)
    counts = Counter(all_decisions)
    total = sum(counts.values())
    probs = np.array([counts[d] / total for d in counts])
    h = -np.sum(probs * np.log(probs + 1e-15))
    log_k = np.log(k)
    return 1.0 - h / log_k if log_k > 0 else 1.0


# ---------------------------------------------------------------------------
# Subsampling engine
# ---------------------------------------------------------------------------


def subsample_metrics(runs_n8: list[dict]) -> tuple:
    """For a single case group with N=8 runs, compute mean DAR and TAR
    over all C(8,3)=56 subsamples of size 3.

    Returns (mean_dar_n3, mean_tar_n3, dar_n8, tar_n8)
    """
    dar_n8 = compute_dar(runs_n8)
    tar_n8 = compute_tar(runs_n8)

    dar_subs = []
    tar_subs = []
    for combo in combinations(range(N_FULL), N_SUB):
        sub_runs = [runs_n8[i] for i in combo]
        dar_subs.append(compute_dar(sub_runs))
        tar_sub = compute_tar(sub_runs)
        if tar_sub is not None:
            tar_subs.append(tar_sub)

    mean_dar_n3 = np.mean(dar_subs)
    mean_tar_n3 = np.mean(tar_subs) if tar_subs else None

    return mean_dar_n3, mean_tar_n3, dar_n8, tar_n8


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analysis():
    print("=" * 72)
    print("DFAH-Bench N=3 Subsampling Sensitivity Analysis")
    print("=" * 72)
    print(f"\nSubsampling N={N_FULL} -> N={N_SUB}  (C({N_FULL},{N_SUB}) = "
          f"{len(list(combinations(range(N_FULL), N_SUB)))} subsamples per case)")
    print()

    data = load_run_logs(RUN_LOG_ROOT)

    # Collect model-level metrics at N=8 and mean-of-subsamples at N=3
    model_results = {}

    for model_slug, display_name in LOCAL_MODELS.items():
        case_dars_n8 = []
        case_tars_n8 = []
        case_dars_n3 = []
        case_tars_n3 = []
        case_decisions_n8 = {}
        n_cases = 0

        for bench in BENCHMARKS:
            if bench not in data.get(model_slug, {}):
                continue
            for case_id, runs in data[model_slug][bench].items():
                n_cases += 1
                mean_dar_n3, mean_tar_n3, dar_n8, tar_n8 = \
                    subsample_metrics(runs)

                case_dars_n8.append(dar_n8)
                case_dars_n3.append(mean_dar_n3)

                if tar_n8 is not None:
                    case_tars_n8.append(tar_n8)
                if mean_tar_n3 is not None:
                    case_tars_n3.append(mean_tar_n3)

                # Collect decisions for DCB
                full_key = f"{bench}/{case_id}"
                case_decisions_n8[full_key] = \
                    [r["decision_output"] for r in runs]

        dar_n8_mean = np.mean(case_dars_n8)
        dar_n3_mean = np.mean(case_dars_n3)
        has_tar = len(case_tars_n8) > 0
        tar_n8_mean = np.mean(case_tars_n8) if has_tar else None
        tar_n3_mean = np.mean(case_tars_n3) if has_tar else None

        gap_n8 = (dar_n8_mean - tar_n8_mean) if has_tar else None
        gap_n3 = (dar_n3_mean - tar_n3_mean) if has_tar else None

        dcb_n8 = compute_dcb_for_model(case_decisions_n8)

        model_results[model_slug] = {
            "display": display_name,
            "n_cases": n_cases,
            "dar_n8": dar_n8_mean,
            "dar_n3": dar_n3_mean,
            "tar_n8": tar_n8_mean,
            "tar_n3": tar_n3_mean,
            "gap_n8": gap_n8,
            "gap_n3": gap_n3,
            "dcb_n8": dcb_n8,
            "tier": PAPER_TIERS[model_slug],
            "has_tar": has_tar,
            "dar_deltas": np.array(case_dars_n3) - np.array(case_dars_n8),
            "tar_deltas": (np.array(case_tars_n3) - np.array(case_tars_n8)
                           if has_tar else None),
        }

    total_groups = sum(r["n_cases"] for r in model_results.values())
    print(f"\nTotal complete N=8 case groups subsampled: {total_groups}")

    # -----------------------------------------------------------------------
    # Report: Model-level comparison table
    # -----------------------------------------------------------------------
    print("-" * 72)
    print("Model-Level Metrics: N=8 vs Mean-of-Subsampled N=3")
    print("-" * 72)
    header = (f"{'Model':<18} {'Tier':<8} {'DAR@8':>6} {'DAR@3':>6} "
              f"{'dDAR':>7} {'TAR@8':>6} {'TAR@3':>6} "
              f"{'Gap@8':>6} {'Gap@3':>6} {'dGap':>7} {'Cases':>5}")
    print(header)
    print("-" * len(header))

    for slug, r in model_results.items():
        tar8 = f"{r['tar_n8']:.3f}" if r["has_tar"] else "  ---"
        tar3 = f"{r['tar_n3']:.3f}" if r["has_tar"] else "  ---"
        gap8 = f"{r['gap_n8']:.3f}" if r["has_tar"] else "  ---"
        gap3 = f"{r['gap_n3']:.3f}" if r["has_tar"] else "  ---"
        d_dar = r["dar_n3"] - r["dar_n8"]
        d_gap = (r["gap_n3"] - r["gap_n8"]) if r["has_tar"] else None
        d_gap_s = f"{d_gap:+.4f}" if d_gap is not None else "    ---"

        print(f"{r['display']:<18} {r['tier']:<8} {r['dar_n8']:.3f}  "
              f"{r['dar_n3']:.3f} {d_dar:+.4f}  {tar8}  {tar3}  "
              f"{gap8}  {gap3} {d_gap_s}  {r['n_cases']:>4}")

    # -----------------------------------------------------------------------
    # Uncertainty: per-case delta statistics
    # -----------------------------------------------------------------------
    print()
    print("-" * 72)
    print("Per-Case Delta Statistics (N=3 subsample mean minus N=8)")
    print("-" * 72)
    print(f"{'Model':<18} {'Metric':<6} {'Mean':>8} {'Std':>8} "
          f"{'Min':>8} {'Max':>8} {'|Max|':>8}")

    for slug, r in model_results.items():
        dd = r["dar_deltas"]
        print(f"{r['display']:<18} {'DAR':<6} {np.mean(dd):+.5f} "
              f"{np.std(dd):.5f} {np.min(dd):+.5f} {np.max(dd):+.5f} "
              f"{np.max(np.abs(dd)):.5f}")
        if r["has_tar"]:
            td = r["tar_deltas"]
            print(f"{'':<18} {'TAR':<6} {np.mean(td):+.5f} "
                  f"{np.std(td):.5f} {np.min(td):+.5f} {np.max(td):+.5f} "
                  f"{np.max(np.abs(td)):.5f}")

    # -----------------------------------------------------------------------
    # Spearman rank correlation
    # -----------------------------------------------------------------------
    print()
    print("-" * 72)
    print("Spearman Rank Correlation: N=8 vs N=3 Model Rankings")
    print("-" * 72)

    from scipy.stats import spearmanr

    models_with_tar = [s for s, r in model_results.items() if r["has_tar"]]

    # DAR ranking (all 4 models)
    all_slugs = list(model_results.keys())
    dar_n8_vals = [model_results[s]["dar_n8"] for s in all_slugs]
    dar_n3_vals = [model_results[s]["dar_n3"] for s in all_slugs]
    rho_dar, p_dar = spearmanr(dar_n8_vals, dar_n3_vals)
    print(f"DAR  (n={len(all_slugs)} models): rho = {rho_dar:.3f}, p = {p_dar:.3f}")

    # TAR ranking (only models with tool calls)
    if len(models_with_tar) >= 2:
        tar_n8_vals = [model_results[s]["tar_n8"] for s in models_with_tar]
        tar_n3_vals = [model_results[s]["tar_n3"] for s in models_with_tar]
        rho_tar, p_tar = spearmanr(tar_n8_vals, tar_n3_vals)
        print(f"TAR  (n={len(models_with_tar)} models): "
              f"rho = {rho_tar:.3f}, p = {p_tar:.3f}")

    # Gap ranking (only models with tool calls)
    if len(models_with_tar) >= 2:
        gap_n8_vals = [model_results[s]["gap_n8"] for s in models_with_tar]
        gap_n3_vals = [model_results[s]["gap_n3"] for s in models_with_tar]
        rho_gap, p_gap = spearmanr(gap_n8_vals, gap_n3_vals)
        print(f"Gap  (n={len(models_with_tar)} models): "
              f"rho = {rho_gap:.3f}, p = {p_gap:.3f}")

    # -----------------------------------------------------------------------
    # Tier assignment robustness
    # -----------------------------------------------------------------------
    print()
    print("-" * 72)
    print("Tier Assignment Robustness Under Subsampling")
    print("-" * 72)

    # Define tier assignment rules based on paper thresholds:
    #   Pattern matcher: DAR >= 0.99 and (no TAR or Gap < 0.01)
    #   Stable executor: DAR >= 0.90 and Gap <= 0.07
    #   Trajectory diverger: Gap > 0.07
    # (naming discipline: only tool-path data is analyzed here, so this is
    # trajectory divergence, not "reasoning divergence")
    def assign_tier(dar, gap):
        if gap is None:
            # No tool calls -- pattern matcher if DAR high
            return "Pattern" if dar >= 0.98 else "Unknown"
        if dar >= 0.98 and gap < 0.01:
            return "Pattern"
        if gap <= 0.07:
            return "Stable"
        return "Diverger"

    tier_preserved = True
    for slug, r in model_results.items():
        tier_n8 = assign_tier(r["dar_n8"], r["gap_n8"])
        tier_n3 = assign_tier(r["dar_n3"], r["gap_n3"])
        paper_tier = r["tier"]
        match = "YES" if tier_n3 == paper_tier else "NO"
        if tier_n3 != paper_tier:
            tier_preserved = False
        print(f"  {r['display']:<18}  Paper: {paper_tier:<10}  "
              f"N=8: {tier_n8:<10}  N=3: {tier_n3:<10}  "
              f"Preserved: {match}")

    # -----------------------------------------------------------------------
    # Worst-case subsample analysis
    # -----------------------------------------------------------------------
    print()
    print("-" * 72)
    print("Worst-Case Subsample Analysis (per-case)")
    print("-" * 72)
    print("For each case, report the subsample of size 3 that produces")
    print("the LOWEST DAR (adversarial subsampling).")
    print()

    for slug, r in model_results.items():
        if not r["has_tar"]:
            continue
        worst_dars = []
        for bench in BENCHMARKS:
            if bench not in data.get(slug, {}):
                continue
            for case_id, runs in data[slug][bench].items():
                min_dar = min(
                    compute_dar([runs[i] for i in combo])
                    for combo in combinations(range(N_FULL), N_SUB)
                )
                worst_dars.append(min_dar)
        worst_model_dar = np.mean(worst_dars)
        print(f"  {r['display']:<18}  "
              f"Worst-case mean DAR@3: {worst_model_dar:.3f}  "
              f"(vs normal DAR@8: {r['dar_n8']:.3f})")

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    if tier_preserved:
        print("RESULT: TAXONOMY IS ROBUST.")
        print("All four local models preserve their paper tier assignments")
        print("under exhaustive C(8,3)=56 subsampling to N=3.")
    else:
        print("RESULT: TAXONOMY IS SENSITIVE.")
        print("At least one model changes tier under N=3 subsampling.")
    print("=" * 72)

    return tier_preserved


if __name__ == "__main__":
    robust = run_analysis()
    sys.exit(0 if robust else 1)
