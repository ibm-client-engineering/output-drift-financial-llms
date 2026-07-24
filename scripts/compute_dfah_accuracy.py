#!/usr/bin/env python3
"""Compute canonical DFAH-Bench accuracy summaries from checked-in run logs.

This script derives the NeurIPS paper's conventional accuracy column directly
from the existing replay corpus. It does not rerun any models; it only reads
the logged model decisions already present under
`econometrics/benchmarks/results/run_logs/`.

Accuracy definition used here:
  1. Group runs by (benchmark, model, case_id)
  2. Take the modal decision for each case group
  3. Mark the case correct when the modal decision matches ground truth
  4. Compute benchmark accuracy as mean case correctness within that benchmark
  5. Compute model-level accuracy as the simple mean of benchmark accuracies

This keeps the aggregation aligned with the paper's task-averaged model table.
Very small partial benchmark slices can be excluded from the model-level mean
via `--min-cases`, which defaults to 10 so that the 3-case Qwen 3.5 portfolio
slice is not treated as a full benchmark.

Usage:
    python scripts/compute_dfah_accuracy.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_LOGS_DIR = REPO_ROOT / "econometrics" / "benchmarks" / "results" / "run_logs"
OUTPUT_DIR = REPO_ROOT / "results"
BENCHMARK_OUTPUT = OUTPUT_DIR / "dfah_accuracy_benchmark_level.csv"
MODEL_OUTPUT = OUTPUT_DIR / "dfah_model_accuracy.csv"

GROUND_TRUTH_SPECS = {
    "compliance": {
        "path": REPO_ROOT / "econometrics" / "benchmarks" / "compliance_triage" / "data" / "alerts.json",
        "key": "alerts",
        "id_key": "alert_id",
    },
    "portfolio": {
        "path": REPO_ROOT / "econometrics" / "benchmarks" / "portfolio_constraint" / "data" / "trades.json",
        "key": "trades",
        "id_key": "trade_id",
    },
    "dataops": {
        "path": REPO_ROOT / "econometrics" / "benchmarks" / "dataops_exception" / "data" / "exceptions.json",
        "key": "exceptions",
        "id_key": "exception_id",
    },
}

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


def load_ground_truth() -> Dict[str, Dict[str, str]]:
    """Load benchmark ground-truth decision labels."""
    gt: Dict[str, Dict[str, str]] = {}
    for benchmark, spec in GROUND_TRUTH_SPECS.items():
        with open(spec["path"]) as f:
            payload = json.load(f)
        items = payload[spec["key"]]
        gt[benchmark] = {
            item[spec["id_key"]]: item["ground_truth"].strip().lower()
            for item in items
        }
    return gt


def iter_run_logs(run_logs_dir: Path) -> Iterable[Tuple[str, str, str, str]]:
    """Yield (benchmark, model, case_id, decision_output) from checked-in logs."""
    for log_file in sorted(run_logs_dir.rglob("case_*_run_*.json")):
        if "_full" in log_file.name:
            continue
        with open(log_file) as f:
            data = json.load(f)
        benchmark = data["benchmark"]
        model = data.get("model") or log_file.parent.name
        case_id = data["case_id"]
        decision = (data.get("decision_output") or "").strip().lower()
        yield benchmark, model, case_id, decision


def build_case_rows(run_logs_dir: Path, ground_truth: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Aggregate raw runs into per-case modal-decision accuracy rows."""
    grouped: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for benchmark, model, case_id, decision in iter_run_logs(run_logs_dir):
        if case_id not in ground_truth.get(benchmark, {}):
            continue
        grouped[(benchmark, model, case_id)].append(decision)

    rows = []
    for (benchmark, model, case_id), decisions in sorted(grouped.items()):
        counts = Counter(decisions)
        modal_decision, modal_count = counts.most_common(1)[0]
        gt = ground_truth[benchmark][case_id]
        rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "model_display": MODEL_DISPLAY.get(model, model),
                "case_id": case_id,
                "n_runs": len(decisions),
                "modal_decision": modal_decision,
                "modal_count": modal_count,
                "modal_decision_accuracy": float(modal_decision == gt),
                "ground_truth": gt,
            }
        )
    return pd.DataFrame(rows)


def aggregate_benchmark_level(case_df: pd.DataFrame) -> pd.DataFrame:
    """Compute benchmark-level modal-decision accuracy per model."""
    rows = []
    for (benchmark, model), grp in case_df.groupby(["benchmark", "model"], sort=True):
        rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "model_display": MODEL_DISPLAY.get(model, model),
                "n_cases": len(grp),
                "total_runs": int(grp["n_runs"].sum()),
                "benchmark_accuracy_pct": 100.0 * grp["modal_decision_accuracy"].mean(),
            }
        )
    return pd.DataFrame(rows)


def aggregate_model_level(benchmark_df: pd.DataFrame, min_cases: int) -> pd.DataFrame:
    """Compute task-averaged model accuracy with a minimum benchmark coverage rule."""
    rows = []
    for model, grp in benchmark_df.groupby("model", sort=False):
        included = grp[grp["n_cases"] >= min_cases].copy()
        excluded = grp[grp["n_cases"] < min_cases].copy()
        if included.empty:
            continue

        rows.append(
            {
                "model": model,
                "model_display": MODEL_DISPLAY.get(model, model),
                "n_benchmarks_included": len(included),
                "included_benchmarks": ",".join(included["benchmark"].tolist()),
                "excluded_benchmarks": ",".join(excluded["benchmark"].tolist()),
                "task_weighted_accuracy_pct": included["benchmark_accuracy_pct"].mean(),
            }
        )

    model_df = pd.DataFrame(rows)
    model_df["sort_key"] = model_df["model"].map(
        {model: idx for idx, model in enumerate(MODEL_ORDER)}
    ).fillna(999)
    model_df = model_df.sort_values(["sort_key", "model"]).drop(columns=["sort_key"])
    return model_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DFAH-Bench accuracy summaries")
    parser.add_argument(
        "--run-logs-dir",
        type=Path,
        default=RUN_LOGS_DIR,
        help="Path to replay run logs (default: checked-in corpus)",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=10,
        help="Minimum observed cases required for a benchmark to count toward model-level Acc",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth()
    case_df = build_case_rows(args.run_logs_dir, ground_truth)
    benchmark_df = aggregate_benchmark_level(case_df)
    model_df = aggregate_model_level(benchmark_df, min_cases=args.min_cases)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(BENCHMARK_OUTPUT, index=False)
    model_df.to_csv(MODEL_OUTPUT, index=False)

    print(f"Wrote benchmark-level accuracy to {BENCHMARK_OUTPUT}")
    print(f"Wrote model-level accuracy to {MODEL_OUTPUT}")
    print(f"\nModel-level task-weighted accuracy (min_cases={args.min_cases}):\n")
    print(
        model_df[
            [
                "model_display",
                "task_weighted_accuracy_pct",
                "included_benchmarks",
                "excluded_benchmarks",
            ]
        ].to_string(index=False, formatters={"task_weighted_accuracy_pct": "{:.3f}".format})
    )


if __name__ == "__main__":
    main()
