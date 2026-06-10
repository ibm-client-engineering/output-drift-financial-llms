#!/usr/bin/env python3
"""Per-task bootstrap confidence intervals for the DAR-TAR gap.

The paper's 18pp model-level headline gap already has a CI
(Appendix~\\ref{app:bootstrap}), but the 27pp DataOps-specific finding for
Claude Sonnet and related task-level callouts are point estimates only.
This script bootstraps CIs at the (benchmark, model) level so the
task-highlight table can report significance alongside the raw values.

Method:
  For each (benchmark, model) pair with trajectory data:
    - Resample case groups with replacement (B = 10,000 times, seed = 42)
    - Compute mean gap per resample
    - Report 2.5/97.5 percentiles

Output:
  results/dfah_task_gap_cis.csv           (table for paper)
  stdout                                   LaTeX snippet + sorted summary

Usage:
    python scripts/compute_task_gap_cis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CASE_CSV = REPO_ROOT / "results" / "dfah_case_level.csv"
OUTPUT_CSV = REPO_ROOT / "results" / "dfah_task_gap_cis.csv"

SEED = 42
N_RESAMPLES = 10000

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

BENCHMARK_DISPLAY = {
    "compliance": "Compliance",
    "dataops":    "DataOps",
    "portfolio":  "Portfolio",
}


def bootstrap_gap(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Percentile bootstrap for the mean of `values`."""
    n = len(values)
    point = float(values.mean())
    if n < 2:
        return point, float("nan"), float("nan")
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    resampled_means = values[idx].mean(axis=1)
    lo, hi = np.quantile(resampled_means, [0.025, 0.975])
    return point, float(lo), float(hi)


def main() -> None:
    df = pd.read_csv(CASE_CSV)
    print(f"Loaded {len(df)} case-level rows from {CASE_CSV}")

    rows = []
    for (benchmark, model), grp in df.groupby(["benchmark", "model"], sort=True):
        eligible = grp[grp["has_trajectory"] & grp["dar_tar_gap"].notna()]
        n = len(eligible)
        if n == 0:
            continue

        # Separate per-(benchmark,model) RNG so each row is reproducible in isolation
        rng = np.random.default_rng(SEED)
        values = eligible["dar_tar_gap"].to_numpy(dtype=float)
        gap, lo, hi = bootstrap_gap(values, rng)

        # Companion DAR and TAR means (unbootstrapped — just context)
        mean_dar = float(eligible["dar"].mean())
        mean_tar = float(eligible["tar"].mean())

        rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "model_display": MODEL_DISPLAY.get(model, model),
                "benchmark_display": BENCHMARK_DISPLAY.get(benchmark, benchmark),
                "n_cases": n,
                "mean_dar": round(mean_dar, 4),
                "mean_tar": round(mean_tar, 4),
                "gap": round(gap, 4),
                "gap_ci_lo": round(lo, 4) if lo == lo else None,
                "gap_ci_hi": round(hi, 4) if hi == hi else None,
                "excludes_zero": (lo > 0) if lo == lo else None,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["benchmark", "gap"], ascending=[True, False])
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} ({len(out)} rows)\n")

    print("Per-task Gap CIs (sorted, benchmark then gap desc):")
    print(out[["benchmark_display", "model_display", "n_cases", "mean_dar",
               "mean_tar", "gap", "gap_ci_lo", "gap_ci_hi", "excludes_zero"]]
          .to_string(index=False))

    # --- LaTeX snippet for the task-highlight rows that matter most ---
    # Pick the rows currently cited in paper §4.2 (Compliance/DataOps/Portfolio
    # for trajectory-diverger models) and print with CIs.
    headline_rows = out[
        out["model"].isin([
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ])
    ].copy()

    print("\n=== LaTeX snippet for task-highlight table (with CIs) ===\n")
    for _, r in headline_rows.iterrows():
        if r["gap_ci_lo"] is None:
            ci = "---"
        else:
            ci = f"[{r['gap_ci_lo']:.3f}, {r['gap_ci_hi']:.3f}]"
        print(
            f"{r['benchmark_display']:<11} & {r['model_display']:<18} & "
            f"{r['mean_dar']:.3f} & {r['mean_tar']:.3f} & "
            f"{r['gap']:.3f} \\tiny{{{ci}}} \\\\"
        )


if __name__ == "__main__":
    main()
