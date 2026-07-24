#!/usr/bin/env python3
"""Compute across-case DCB (paper-accurate definition).

The paper defines DCB as concentration of the model's decision distribution
*across different test cases within a task*:

    DCB = 1 - H(p) / log(K)
    where p = empirical distribution of per-case modal decisions
    across the 50 test cases of a benchmark.

The prior implementation in compute_dfah_metrics.py computed DCB *within each
case* (over N replays) and averaged, which is tautologically close to DAR
and does not match the paper's stated definition. This script computes the
correct across-case DCB and writes it as standalone CSV outputs that Table 1
and Figure 2 can consume.

Aggregation matches the paper's task-averaged model table: benchmark-level
DCB is computed per (benchmark, model); the model-level value is the simple
mean of benchmark-level DCBs over benchmarks with at least `min_cases`
observed cases (default 10).

Usage:
    python scripts/compute_dcb_across_case.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_LOGS_DIR = REPO_ROOT / "econometrics" / "benchmarks" / "results" / "run_logs"
OUTPUT_DIR = REPO_ROOT / "results"
BENCH_OUTPUT = OUTPUT_DIR / "dfah_dcb_across_case_benchmark.csv"
MODEL_OUTPUT = OUTPUT_DIR / "dfah_dcb_across_case_model.csv"

K = 3  # all three benchmarks have K=3 closed ontologies

MODEL_DISPLAY = {
    "qwen3.5_latest": "Qwen 3.5",
    "qwen2.5_7b-instruct": "Qwen 2.5 7B",
    "granite3.3_latest": "Granite 3.3",
    "mistral_7b": "Mistral 7B",
    "gemma4_latest": "Gemma 4",
    "gpt-oss_20b": "GPT-OSS 20B",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "claude-opus-4-20250514": "Claude Opus 4",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "deepseek-r1_8b": "DeepSeek-R1 8B",
}

MODEL_ORDER = [
    "qwen3.5_latest", "gemma4_latest", "qwen2.5_7b-instruct",
    "granite3.3_latest", "mistral_7b", "gpt-oss_20b",
    "gemini-2.0-flash", "gemini-2.5-pro",
    "claude-opus-4-20250514", "claude-sonnet-4-20250514",
]


def dcb_from_counts(counts: Dict[str, int], k: int = K) -> float:
    total = sum(counts.values())
    if total == 0 or k <= 1:
        return float("nan")
    probs = np.array([v / total for v in counts.values()], dtype=float)
    probs = probs[probs > 0]
    H = float(-np.sum(probs * np.log(probs)))
    return 1.0 - H / np.log(k)


def per_case_modal(decisions: List[str]) -> str:
    return Counter(decisions).most_common(1)[0][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cases", type=int, default=10,
                    help="Minimum observed cases for a benchmark to count toward model-level DCB")
    args = ap.parse_args()

    # Group per-run decisions by (benchmark, model, case_id)
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

    # Benchmark-level: distribution of per-case modal decisions
    bench_rows = []
    for (benchmark, model), _ in sorted(
        {(b, m): None for (b, m, _) in grouped}.items()
    ):
        case_decisions = {
            c: d for (b, m, c), d in grouped.items() if b == benchmark and m == model
        }
        modal_decisions = [per_case_modal(d) for c, d in case_decisions.items()]
        modal_counts = Counter(modal_decisions)
        dcb_ac = dcb_from_counts(modal_counts)
        bench_rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "model_display": MODEL_DISPLAY.get(model, model),
                "n_cases": len(case_decisions),
                "dcb_across_case": dcb_ac,
                "modal_distribution": json.dumps(dict(modal_counts)),
            }
        )

    bench_df = pd.DataFrame(bench_rows).sort_values(["benchmark", "model"])

    # Model-level: mean of benchmark DCBs restricted to benchmarks with enough cases
    model_rows = []
    for model in MODEL_ORDER:
        mdf = bench_df[bench_df["model"] == model]
        included = mdf[mdf["n_cases"] >= args.min_cases]
        excluded = mdf[mdf["n_cases"] < args.min_cases]
        if included.empty:
            continue
        model_rows.append(
            {
                "model": model,
                "model_display": MODEL_DISPLAY.get(model, model),
                "n_benchmarks_included": len(included),
                "included_benchmarks": ",".join(sorted(included["benchmark"].tolist())),
                "excluded_benchmarks": ",".join(sorted(excluded["benchmark"].tolist())) or "",
                "mean_dcb_across_case": included["dcb_across_case"].mean(),
                "min_bench_dcb": included["dcb_across_case"].min(),
                "max_bench_dcb": included["dcb_across_case"].max(),
            }
        )
    model_df = pd.DataFrame(model_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bench_df.to_csv(BENCH_OUTPUT, index=False)
    model_df.to_csv(MODEL_OUTPUT, index=False)
    print(f"Wrote {BENCH_OUTPUT}")
    print(f"Wrote {MODEL_OUTPUT}\n")

    print("Benchmark-level across-case DCB:")
    print(bench_df.to_string(index=False))

    print("\nModel-level across-case DCB (min_cases>=10 rule):")
    print(model_df.to_string(index=False))

    # GT baseline: DCB of the ground-truth label distributions themselves
    gt_specs = [
        ("compliance", REPO_ROOT / "econometrics" / "benchmarks" / "compliance_triage" / "data" / "alerts.json"),
        ("portfolio", REPO_ROOT / "econometrics" / "benchmarks" / "portfolio_constraint" / "data" / "trades.json"),
        ("dataops", REPO_ROOT / "econometrics" / "benchmarks" / "dataops_exception" / "data" / "exceptions.json"),
    ]
    gt_dcbs = []
    for benchmark, path in gt_specs:
        dist = json.loads(path.read_text())["metadata"]["ground_truth_distribution"]
        gt_dcbs.append((benchmark, dcb_from_counts(dist)))
    print("\nGround-truth reference DCB (for comparison):")
    for bench, v in gt_dcbs:
        print(f"  {bench:<12s} {v:.4f}")
    print(f"  task-avg:    {np.mean([v for _, v in gt_dcbs]):.4f}")


if __name__ == "__main__":
    main()
