#!/usr/bin/env python3
"""Compute Fleiss' kappa for within-case decision agreement.

Reviewer concern: DAR is the within-case modal-match rate. With K=3 closed
ontologies, chance agreement is 33%, so a 95% DAR may be partially chance-
inflated. Cohen/Fleiss kappa corrects for chance agreement using the model's
own marginal preference distribution. This is the right metric: it asks
"given that this model picks 'escalate' p% of the time anyway, how much MORE
agreement is there within a single case than that marginal would predict?"

Method (per benchmark, per model):
  1. Marginal p_k = fraction of all decisions across cases for label k
  2. Expected agreement p_e = sum_k p_k^2 (two-rater chance agreement)
  3. Per-case observed agreement
       p_o(case) = sum_k n_k(n_k-1) / (N(N-1))     for N >= 2 runs
  4. Mean p_o across the model's case groups in that benchmark
  5. kappa = (mean_p_o - p_e) / (1 - p_e)

Then model-level kappa = simple mean of benchmark-level kappas (matching the
paper's task-averaged aggregation).

Reads the existing run logs; runs no models.

Usage:
    python scripts/compute_kappa.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_LOGS_DIR = REPO_ROOT / "econometrics" / "benchmarks" / "results" / "run_logs"
OUTPUT_DIR = REPO_ROOT / "results"
BENCHMARK_OUTPUT = OUTPUT_DIR / "dfah_kappa_benchmark_level.csv"
MODEL_OUTPUT = OUTPUT_DIR / "dfah_kappa_model_level.csv"

MODEL_DISPLAY = {
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

MODEL_ORDER = [
    "qwen3.5_latest", "gemma4_latest", "qwen2.5_7b-instruct",
    "granite3.3_latest", "mistral_7b", "gpt-oss_20b",
    "gemini-2.0-flash", "gemini-2.5-pro",
    "claude-opus-4-20250514", "claude-sonnet-4-20250514",
]


def load_decisions() -> Dict[Tuple[str, str, str], List[str]]:
    """Group decisions by (benchmark, model, case_id)."""
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
        if not decision:
            continue
        grouped[(benchmark, model, case_id)].append(decision)
    return grouped


def kappa_for_group(
    case_decisions: Dict[str, List[str]],
) -> Tuple[float, float, float, int, int]:
    """Compute Fleiss-style kappa from {case_id: [decision, ...]}.

    Returns (kappa, mean_p_o, p_e, n_cases_used, total_runs).
    """
    # Marginal distribution across all runs in this (model, benchmark)
    all_runs = [d for ds in case_decisions.values() for d in ds]
    total = len(all_runs)
    marginals = Counter(all_runs)
    p = {k: v / total for k, v in marginals.items()}
    p_e = sum(v * v for v in p.values())

    # Per-case observed agreement (only N >= 2)
    p_o_values = []
    for case_id, ds in case_decisions.items():
        n = len(ds)
        if n < 2:
            continue
        counts = Counter(ds)
        agree_pairs = sum(c * (c - 1) for c in counts.values())
        total_pairs = n * (n - 1)
        p_o_values.append(agree_pairs / total_pairs)

    if not p_o_values:
        return float("nan"), float("nan"), p_e, 0, total

    mean_p_o = sum(p_o_values) / len(p_o_values)
    if p_e >= 1.0:
        kappa = float("nan")  # Marginal is degenerate (single label)
    else:
        kappa = (mean_p_o - p_e) / (1.0 - p_e)
    return kappa, mean_p_o, p_e, len(p_o_values), total


def main() -> None:
    grouped = load_decisions()

    # Reorganize: {(benchmark, model): {case_id: [decisions]}}
    by_model_bench: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(dict)
    for (benchmark, model, case_id), decisions in grouped.items():
        by_model_bench[(benchmark, model)][case_id] = decisions

    # Per-benchmark rows
    bench_rows = []
    for (benchmark, model), case_decisions in sorted(by_model_bench.items()):
        kappa, mean_p_o, p_e, n_cases, total = kappa_for_group(case_decisions)
        bench_rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "model_display": MODEL_DISPLAY.get(model, model),
                "n_cases": n_cases,
                "total_runs": total,
                "p_e_chance_agreement": round(p_e, 4),
                "mean_p_o_observed_agreement": round(mean_p_o, 4),
                "kappa": round(kappa, 4) if kappa == kappa else None,
            }
        )

    bench_df = pd.DataFrame(bench_rows)

    # Model-level: simple mean of benchmark kappas (matches paper aggregation)
    model_rows = []
    for model in MODEL_ORDER:
        mdf = bench_df[bench_df["model"] == model]
        if mdf.empty:
            continue
        kappas = mdf["kappa"].dropna()
        model_rows.append(
            {
                "model": model,
                "model_display": MODEL_DISPLAY.get(model, model),
                "n_benchmarks": len(mdf),
                "n_benchmarks_with_kappa": len(kappas),
                "mean_kappa": round(kappas.mean(), 4) if len(kappas) else None,
                "min_kappa": round(kappas.min(), 4) if len(kappas) else None,
                "max_kappa": round(kappas.max(), 4) if len(kappas) else None,
            }
        )
    model_df = pd.DataFrame(model_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bench_df.to_csv(BENCHMARK_OUTPUT, index=False)
    model_df.to_csv(MODEL_OUTPUT, index=False)
    print(f"Wrote {BENCHMARK_OUTPUT}")
    print(f"Wrote {MODEL_OUTPUT}\n")

    print("Benchmark-level kappa:")
    print(bench_df.to_string(index=False))

    print("\nModel-level kappa (task-averaged):")
    print(model_df.to_string(index=False))


if __name__ == "__main__":
    main()
