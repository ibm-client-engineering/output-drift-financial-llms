#!/usr/bin/env python3
"""Compute ground-truth reference baselines for DFAH-Bench.

Reviewer concern: a model's high DCB might just reflect a skewed ground-truth
label distribution rather than pattern-matching behavior. This script computes
the DCB of the ground-truth distribution itself per benchmark, plus the
accuracy of an "always-modal" oracle baseline. Both are reported as reference
rows alongside the per-model results.

Reads:
  econometrics/benchmarks/{compliance_triage,portfolio_constraint,dataops_exception}/data/*.json

Writes:
  results/dfah_gt_baselines.csv

Usage:
    python scripts/compute_gt_baselines.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench.metrics.dcb import compute_dcb  # noqa: E402

OUTPUT_CSV = REPO_ROOT / "results" / "dfah_gt_baselines.csv"

GT_SPECS = {
    "compliance": REPO_ROOT / "econometrics" / "benchmarks" / "compliance_triage" / "data" / "alerts.json",
    "portfolio":  REPO_ROOT / "econometrics" / "benchmarks" / "portfolio_constraint" / "data" / "trades.json",
    "dataops":    REPO_ROOT / "econometrics" / "benchmarks" / "dataops_exception" / "data" / "exceptions.json",
}


def main() -> None:
    rows = []
    for benchmark, path in GT_SPECS.items():
        payload = json.loads(path.read_text())
        gt_dist = payload["metadata"]["ground_truth_distribution"]
        n_cases = sum(gt_dist.values())

        # Reconstruct the per-case label list from the distribution
        labels = []
        for label, count in gt_dist.items():
            labels.extend([label] * count)

        result = compute_dcb(labels, benchmark=benchmark)

        # Always-modal accuracy: an oracle that always predicts the most frequent
        # ground-truth label achieves this accuracy. This is the floor any
        # learned model should beat.
        modal_label, modal_count = max(gt_dist.items(), key=lambda kv: kv[1])
        majority_acc = modal_count / n_cases

        # Random-guess accuracy for K=3 closed ontology
        chance_acc = 1.0 / result.k_categories

        rows.append(
            {
                "benchmark": benchmark,
                "n_cases": n_cases,
                "k": result.k_categories,
                "gt_dcb": round(result.dcb, 4),
                "gt_entropy": round(result.entropy, 4),
                "max_entropy": round(result.max_entropy, 4),
                "modal_label": modal_label,
                "majority_acc_pct": round(majority_acc * 100, 2),
                "chance_acc_pct": round(chance_acc * 100, 2),
                "distribution": json.dumps(gt_dist),
            }
        )

    df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}\n")

    print("Ground-truth baselines (reference rows for Table 1):\n")
    print(df.to_string(index=False))

    # Macro reference: simple mean across the three benchmarks, matching the
    # paper's task-averaged Acc and DCB aggregation.
    print("\nTask-averaged GT reference (matches paper's Acc/DCB aggregation):")
    print(f"  GT DCB        = {df['gt_dcb'].mean():.4f}")
    print(f"  Majority Acc  = {df['majority_acc_pct'].mean():.2f}%")
    print(f"  Chance   Acc  = {df['chance_acc_pct'].mean():.2f}%")


if __name__ == "__main__":
    main()
