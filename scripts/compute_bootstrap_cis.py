#!/usr/bin/env python3
"""Compute 95% bootstrap confidence intervals for model-level DFAH-Bench metrics.

Reads case-level results and bootstraps CIs for model-level metrics while
preserving the paper's task-averaged aggregation:

  1. Resample case groups with replacement *within each benchmark*
  2. Recompute the benchmark-level mean for that metric
  3. Average the benchmark means for the model

This keeps the appendix aligned with the main model table, which reports
equal-weighted benchmark means rather than case-weighted means.

DAR/TAR/Gap/ECD are scalar case-level metrics. Table 1 DCB and kappa require
metric-specific recomputation inside each bootstrap draw:

  - DCB resamples per-case modal decisions within each benchmark, recomputes
    across-case entropy concentration, then task-averages benchmarks with at
    least 10 observed cases.
  - kappa resamples case decision lists within each benchmark and recomputes
    both observed agreement and marginal chance agreement for that draw.

Usage:
    python scripts/compute_bootstrap_cis.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CASE_CSV = RESULTS_DIR / "dfah_case_level.csv"
OUTPUT_CSV = RESULTS_DIR / "dfah_model_cis.csv"
RUN_LOGS_DIR = (
    Path(__file__).resolve().parent.parent
    / "econometrics"
    / "benchmarks"
    / "results"
    / "run_logs"
)

SEED = 42
N_RESAMPLES = 10000

# Display-friendly model names
MODEL_NAMES = {
    "qwen3.5_latest": "Qwen 3.5",
    "qwen2.5_7b-instruct": "Qwen 2.5 7B",
    "granite3.3_latest": "Granite 3.3",
    "mistral_7b": "Mistral 7B",
    "gemma4_latest": "Gemma 4",
    "gpt-oss_20b": "GPT-OSS 20B",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "claude-opus-4-20250514": "Claude Opus 4.5",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "deepseek-r1_8b": "DeepSeek-R1 8B",
}

# Presentation order matches Table 1.
MODEL_ORDER = [
    "qwen3.5_latest",
    "gemma4_latest",
    "qwen2.5_7b-instruct",
    "granite3.3_latest",
    "mistral_7b",
    "gpt-oss_20b",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
]


METRIC_SPECS = {
    "DAR": {
        "value_col": "dar",
        "eligibility_mask": lambda df: df["dar"].notna(),
    },
    "TAR": {
        "value_col": "tar",
        "eligibility_mask": lambda df: df["has_trajectory"] & df["tar"].notna(),
    },
    "Gap": {
        "value_col": "dar_tar_gap",
        "eligibility_mask": lambda df: df["has_trajectory"] & df["dar_tar_gap"].notna(),
    },
    "ECD": {
        "value_col": "ecd",
        "eligibility_mask": lambda df: df["has_evidence"] & df["ecd"].notna(),
    },
}

K = 3


def benchmark_aware_bootstrap(
    mdf: pd.DataFrame,
    metric: str,
) -> tuple[float, float, float, int, int]:
    """Bootstrap a task-averaged model metric from case-level rows.

    Returns (point, ci_lo, ci_hi, n_case_groups, n_benchmarks).
    """
    spec = METRIC_SPECS[metric]
    eligible = mdf.loc[spec["eligibility_mask"](mdf), ["benchmark", spec["value_col"]]].copy()
    eligible = eligible.rename(columns={spec["value_col"]: "value"})

    if eligible.empty:
        return np.nan, np.nan, np.nan, 0, 0

    per_benchmark = {
        benchmark: grp["value"].to_numpy(dtype=float)
        for benchmark, grp in eligible.groupby("benchmark", sort=True)
    }
    benchmark_means = [values.mean() for values in per_benchmark.values()]
    point = float(np.mean(benchmark_means))
    n_case_groups = int(len(eligible))
    n_benchmarks = int(len(per_benchmark))

    if n_case_groups < 2:
        return point, np.nan, np.nan, n_case_groups, n_benchmarks

    rng = np.random.default_rng(SEED)
    resampled = np.zeros(N_RESAMPLES, dtype=float)
    for values in per_benchmark.values():
        n = len(values)
        sampled_idx = rng.integers(0, n, size=(N_RESAMPLES, n))
        resampled += values[sampled_idx].mean(axis=1)
    resampled /= n_benchmarks

    lo, hi = np.quantile(resampled, [0.025, 0.975])
    return point, float(lo), float(hi), n_case_groups, n_benchmarks


def dcb_from_labels(labels: np.ndarray, k: int = K) -> float:
    """Compute entropy-normalized concentration from a label vector."""
    if len(labels) == 0 or k <= 1:
        return float("nan")
    counts = Counter(labels)
    probs = np.array([v / len(labels) for v in counts.values()], dtype=float)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs)))
    return 1.0 - entropy / np.log(k)


def bootstrap_across_case_dcb(
    model_id: str,
    decision_groups: Dict[Tuple[str, str], List[List[str]]],
    min_cases: int = 10,
) -> tuple[float, float, float, int, int]:
    """Bootstrap Table 1 DCB from per-case modal decisions across cases."""
    per_benchmark = {
        benchmark: np.array(
            [Counter(case_decisions).most_common(1)[0][0] for case_decisions in case_lists],
            dtype=str,
        )
        for (benchmark, model), case_lists in decision_groups.items()
        if model == model_id and len(case_lists) >= min_cases
    }
    if not per_benchmark:
        return np.nan, np.nan, np.nan, 0, 0

    benchmark_dcbs = [dcb_from_labels(labels) for labels in per_benchmark.values()]
    point = float(np.mean(benchmark_dcbs))
    n_case_groups = int(sum(len(labels) for labels in per_benchmark.values()))
    n_benchmarks = int(len(per_benchmark))

    if n_case_groups < 2:
        return point, np.nan, np.nan, n_case_groups, n_benchmarks

    rng = np.random.default_rng(SEED)
    resampled = np.zeros(N_RESAMPLES, dtype=float)
    for labels in per_benchmark.values():
        n = len(labels)
        sampled_idx = rng.integers(0, n, size=(N_RESAMPLES, n))
        resampled += np.array(
            [dcb_from_labels(labels[idx]) for idx in sampled_idx],
            dtype=float,
        )
    resampled /= n_benchmarks

    lo, hi = np.nanquantile(resampled, [0.025, 0.975])
    return point, float(lo), float(hi), n_case_groups, n_benchmarks


def load_decision_groups(min_runs: int = 2) -> Dict[Tuple[str, str], List[List[str]]]:
    """Load run-log decisions grouped as {(benchmark, model): case decision lists}."""
    grouped: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for log_file in sorted(RUN_LOGS_DIR.rglob("case_*_run_*.json")):
        if "_full" in log_file.name:
            continue
        with open(log_file) as f:
            data = json.load(f)
        benchmark = data["benchmark"]
        model = data.get("model") or log_file.parent.name
        case_id = data["case_id"]
        decision = (data.get("decision_output") or "").strip().lower()
        if decision:
            grouped[(benchmark, model, case_id)].append(decision)

    by_benchmark_model: Dict[Tuple[str, str], List[List[str]]] = defaultdict(list)
    for (benchmark, model, _case_id), decisions in grouped.items():
        if len(decisions) >= min_runs:
            by_benchmark_model[(benchmark, model)].append(decisions)
    return by_benchmark_model


def kappa_from_case_lists(case_lists: List[List[str]]) -> float:
    """Compute Fleiss-style kappa from a list of per-case decision lists."""
    all_runs = [decision for decisions in case_lists for decision in decisions]
    if not all_runs:
        return float("nan")

    total = len(all_runs)
    marginals = Counter(all_runs)
    p_e = sum((count / total) ** 2 for count in marginals.values())

    p_o_values = []
    for decisions in case_lists:
        n = len(decisions)
        if n < 2:
            continue
        counts = Counter(decisions)
        agree_pairs = sum(count * (count - 1) for count in counts.values())
        p_o_values.append(agree_pairs / (n * (n - 1)))

    if not p_o_values or p_e >= 1.0:
        return float("nan")

    mean_p_o = sum(p_o_values) / len(p_o_values)
    return (mean_p_o - p_e) / (1.0 - p_e)


def bootstrap_kappa(
    model_id: str,
    decision_groups: Dict[Tuple[str, str], List[List[str]]],
) -> tuple[float, float, float, int, int]:
    """Bootstrap task-averaged kappa from run-log decision lists."""
    per_benchmark = {
        benchmark: case_lists
        for (benchmark, model), case_lists in decision_groups.items()
        if model == model_id and case_lists
    }
    if not per_benchmark:
        return np.nan, np.nan, np.nan, 0, 0

    benchmark_kappas = [kappa_from_case_lists(case_lists) for case_lists in per_benchmark.values()]
    benchmark_kappas = [value for value in benchmark_kappas if not np.isnan(value)]
    if not benchmark_kappas:
        return np.nan, np.nan, np.nan, 0, 0

    point = float(np.mean(benchmark_kappas))
    n_case_groups = int(sum(len(case_lists) for case_lists in per_benchmark.values()))
    n_benchmarks = int(len(benchmark_kappas))

    if n_case_groups < 2:
        return point, np.nan, np.nan, n_case_groups, n_benchmarks

    rng = np.random.default_rng(SEED)
    resampled = np.zeros(N_RESAMPLES, dtype=float)
    valid_counts = np.zeros(N_RESAMPLES, dtype=int)

    for case_lists in per_benchmark.values():
        n = len(case_lists)
        sampled_idx = rng.integers(0, n, size=(N_RESAMPLES, n))
        values = np.array(
            [kappa_from_case_lists([case_lists[i] for i in idx]) for idx in sampled_idx],
            dtype=float,
        )
        valid = ~np.isnan(values)
        resampled[valid] += values[valid]
        valid_counts[valid] += 1

    valid_draws = valid_counts > 0
    resampled = resampled[valid_draws] / valid_counts[valid_draws]
    if len(resampled) == 0:
        return point, np.nan, np.nan, n_case_groups, n_benchmarks

    lo, hi = np.nanquantile(resampled, [0.025, 0.975])
    return point, float(lo), float(hi), n_case_groups, n_benchmarks


def main():
    df = pd.read_csv(CASE_CSV)
    print(f"Loaded {len(df)} case-level rows from {CASE_CSV}")
    kappa_decision_groups = load_decision_groups(min_runs=2)
    dcb_decision_groups = load_decision_groups(min_runs=1)

    rows = []

    for model_id in MODEL_ORDER:
        mdf = df[df["model"] == model_id]
        if mdf.empty:
            continue

        name = MODEL_NAMES.get(model_id, model_id)

        for metric in ["DAR", "TAR", "Gap", "ECD"]:
            pt, lo, hi, n, n_benchmarks = benchmark_aware_bootstrap(mdf, metric)
            rows.append(
                {
                    "model": name,
                    "model_id": model_id,
                    "metric": metric,
                    "point": pt,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n": n,
                    "n_benchmarks": n_benchmarks,
                    "aggregation": "benchmark_mean",
                }
            )

        pt, lo, hi, n, n_benchmarks = bootstrap_kappa(model_id, kappa_decision_groups)
        rows.append(
            {
                "model": name,
                "model_id": model_id,
                "metric": "Kappa",
                "point": pt,
                "ci_lo": lo,
                "ci_hi": hi,
                "n": n,
                "n_benchmarks": n_benchmarks,
                "aggregation": "benchmark_mean",
            }
        )

        pt, lo, hi, n, n_benchmarks = bootstrap_across_case_dcb(model_id, dcb_decision_groups)
        rows.append(
            {
                "model": name,
                "model_id": model_id,
                "metric": "DCB",
                "point": pt,
                "ci_lo": lo,
                "ci_hi": hi,
                "n": n,
                "n_benchmarks": n_benchmarks,
                "aggregation": "benchmark_mean_min_cases_10",
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(result_df)} rows to {OUTPUT_CSV}")

    # --- Print summary table ---
    print("\n=== Model-Level 95% Bootstrap CIs ===\n")
    print(f"{'Model':<22} {'Metric':<6} {'Point':>7} {'95% CI':>18} {'n':>5}")
    print("-" * 62)
    for _, row in result_df.iterrows():
        if np.isnan(row["ci_lo"]):
            ci_str = "---"
        else:
            ci_str = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
        print(f"{row['model']:<22} {row['metric']:<6} {row['point']:>7.3f} {ci_str:>18} {int(row['n']):>5}")

    # --- LaTeX table for appendix ---
    print("\n\n=== LaTeX Table (copy to paper) ===\n")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{95\% bootstrap confidence intervals for task-averaged model-level metrics")
    print(r"($B = 10{,}000$ resamples, seed $= 42$). Case groups are resampled within each")
    print(r"benchmark, then benchmark means are averaged so the appendix matches Table~1.}")
    print(r"\label{tab:bootstrap_cis}")
    print(r"\small")
    print(r"\begin{tabular}{@{}lrrrrrr@{}}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{DAR} & \textbf{$\kappa$} & \textbf{TAR} & \textbf{Gap} & \textbf{DCB} & \textbf{ECD} \\")
    print(r"\midrule")

    for model_id in MODEL_ORDER:
        name = MODEL_NAMES.get(model_id, model_id)
        mrows = result_df[result_df["model_id"] == model_id]
        if mrows.empty:
            continue

        cells = []
        for metric in ["DAR", "Kappa", "TAR", "Gap", "DCB", "ECD"]:
            mr = mrows[mrows["metric"] == metric]
            if mr.empty or np.isnan(mr.iloc[0]["ci_lo"]):
                cells.append("---")
            else:
                r = mr.iloc[0]

                def fmt(v):
                    if v < 0:
                        return f"${{-}}${abs(v):.3f}"
                    return f"{v:.3f}"

                cells.append(
                    f"{fmt(r['point'])} "
                    f"\\tiny{{[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]}}"
                )

        latex_name = name.replace("~", "~")
        print(f"  {latex_name} & {' & '.join(cells)} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
