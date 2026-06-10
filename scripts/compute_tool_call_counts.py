#!/usr/bin/env python3
"""Compute average tool calls per run per model.

Reviewer concern: Table 1 shows dashes for TAR/ECD for several models (Granite,
Mistral, DeepSeek) without explaining why. The answer is simple: those models
emit no tool calls at all, so trajectory/evidence channels are empty. Surfacing
an "avg tool calls/run" column makes this self-explanatory and also supports
the Gemma-4 observation that active tool use does not preclude pattern matching.

Reads the existing run logs; runs no models.

Usage:
    python scripts/compute_tool_call_counts.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_LOGS_DIR = REPO_ROOT / "econometrics" / "benchmarks" / "results" / "run_logs"
OUTPUT_DIR = REPO_ROOT / "results"
BENCHMARK_OUTPUT = OUTPUT_DIR / "dfah_tool_call_counts_benchmark.csv"
MODEL_OUTPUT = OUTPUT_DIR / "dfah_tool_call_counts_model.csv"

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


def tool_call_count(data: dict) -> int:
    """Extract the number of tool calls for a run, tolerant to shape variation."""
    seq = data.get("tool_sequence")
    if isinstance(seq, list):
        return len(seq)
    calls = data.get("tool_calls")
    if isinstance(calls, list):
        return len(calls)
    return 0


def main() -> None:
    counts: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for log_file in sorted(RUN_LOGS_DIR.rglob("case_*_run_*.json")):
        if "_full" in log_file.name:
            continue
        with open(log_file) as f:
            data = json.load(f)
        benchmark = data["benchmark"]
        model = data.get("model") or log_file.parent.name
        counts[(benchmark, model)].append(tool_call_count(data))

    # Benchmark-level rows
    bench_rows = []
    for (benchmark, model), values in sorted(counts.items()):
        n = len(values)
        zero = sum(1 for v in values if v == 0)
        mean = sum(values) / n if n else float("nan")
        bench_rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "model_display": MODEL_DISPLAY.get(model, model),
                "n_runs": n,
                "mean_tool_calls": round(mean, 2),
                "pct_runs_with_zero_calls": round(100.0 * zero / n, 1) if n else None,
            }
        )
    bench_df = pd.DataFrame(bench_rows)

    # Model-level (simple mean of benchmark means, matching paper aggregation)
    model_rows = []
    for model in MODEL_ORDER:
        mdf = bench_df[bench_df["model"] == model]
        if mdf.empty:
            continue
        model_rows.append(
            {
                "model": model,
                "model_display": MODEL_DISPLAY.get(model, model),
                "n_benchmarks": len(mdf),
                "total_runs": int(mdf["n_runs"].sum()),
                "mean_tool_calls_per_run": round(mdf["mean_tool_calls"].mean(), 2),
                "max_benchmark_mean": round(mdf["mean_tool_calls"].max(), 2),
                "min_benchmark_mean": round(mdf["mean_tool_calls"].min(), 2),
                "pct_zero_tool_runs": round(mdf["pct_runs_with_zero_calls"].mean(), 1),
            }
        )
    model_df = pd.DataFrame(model_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bench_df.to_csv(BENCHMARK_OUTPUT, index=False)
    model_df.to_csv(MODEL_OUTPUT, index=False)
    print(f"Wrote {BENCHMARK_OUTPUT}")
    print(f"Wrote {MODEL_OUTPUT}\n")

    print("Benchmark-level:")
    print(bench_df.to_string(index=False))
    print("\nModel-level (task-averaged):")
    print(model_df.to_string(index=False))


if __name__ == "__main__":
    main()
