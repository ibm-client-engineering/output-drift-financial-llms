#!/usr/bin/env python3
"""Priority 2 — Perturbation Experiment (v2: Compliance benchmark).

Tests whether DFAH-Bench trajectory divergence predicts decision fragility.

Design:
  - HIGH group: 10 Compliance cases where Claude Sonnet had DAR=1.0 but TAR < 0.9
    (same decision, different tool paths in the original corpus)
  - LOW group: 10 Compliance cases where Claude Sonnet had DAR=1.0 and TAR = 1.0
    (same decision, same tool path every time)

  Both groups have genuine decision variation (investigate, escalate, dismiss)
  across cases — NOT pattern matching.

Perturbation:
  - Change the transaction amount by +15% (enough to be noticed, not enough
    to cross the $50K risk-score threshold in most cases)
  - This preserves the ground truth for most cases since compliance decisions
    depend on sanctions, KYC status, and country risk, not primarily on amount.

Protocol:
  - Run each perturbed case 3 times with Claude Sonnet (claude-sonnet-4-20250514)
  - Record decision, tool sequence, tool count
  - Measure decision flip rate vs original modal decision from the corpus

Hypothesis:
  High-divergence cases flip decisions more often under perturbation than
  low-divergence cases, even though outcome-only evaluation treats both groups
  as equally "stable" (both have DAR = 1.0).

v1 result (DataOps): Both groups flipped 100% under the shared "quarantine"
  cue, so the perturbation did not distinguish the groups. The v2 experiment
  switches to Compliance, where decisions are conditioned on several rules.
"""

import copy
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent / "econometrics" / "benchmarks"
    ),
)

from run_agentic_benchmark import (
    COMPLIANCE_TOOLS_ANTHROPIC,
    execute_tool,
    load_alerts,
    run_agent_anthropic,
)

# ---------------------------------------------------------------------------
# Case selection (from P1 results: Compliance / Claude Sonnet)
# ---------------------------------------------------------------------------

# HIGH divergence: DAR=1.0, TAR < 0.9 (2-3 unique tool sequences across 3 runs)
HIGH_CASES = [
    "TXN-2025-009",  # TAR=0.333, modal=investigate
    "TXN-2025-018",  # TAR=0.333, modal=investigate
    "TXN-2025-046",  # TAR=0.333, modal=investigate
    "TXN-2025-004",  # TAR=0.667, modal=escalate
    "TXN-2025-043",  # TAR=0.667, modal=escalate
    "TXN-2025-042",  # TAR=0.667, modal=investigate
    "TXN-2025-040",  # TAR=0.667, modal=investigate
    "TXN-2025-038",  # TAR=0.667, modal=investigate
    "TXN-2025-030",  # TAR=0.667, modal=investigate
    "TXN-2025-027",  # TAR=0.667, modal=escalate
]

# LOW divergence: DAR=1.0, TAR=1.0 (identical tool path all 3 runs)
LOW_CASES = [
    "TXN-2025-002",  # TAR=1.0, modal=escalate
    "TXN-2025-005",  # TAR=1.0, modal=investigate
    "TXN-2025-049",  # TAR=1.0, modal=escalate
    "TXN-2025-048",  # TAR=1.0, modal=investigate
    "TXN-2025-044",  # TAR=1.0, modal=investigate
    "TXN-2025-041",  # TAR=1.0, modal=escalate
    "TXN-2025-039",  # TAR=1.0, modal=escalate
    "TXN-2025-037",  # TAR=1.0, modal=escalate
    "TXN-2025-036",  # TAR=1.0, modal=investigate
    "TXN-2025-035",  # TAR=1.0, modal=escalate
]

ORIGINAL_DECISIONS: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------


def perturb_alert(alert: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict]:
    """Perturb a compliance alert by changing the transaction amount +15%."""
    perturbed = copy.deepcopy(alert)
    original_amount = perturbed["amount"]
    perturbed["amount"] = round(original_amount * 1.15, 2)
    meta = {
        "type": "amount_increase_15pct",
        "field": "amount",
        "original": original_amount,
        "perturbed": perturbed["amount"],
    }
    return perturbed, meta


# ---------------------------------------------------------------------------
# Load original decisions from corpus
# ---------------------------------------------------------------------------


def load_original_decisions(case_ids: List[str]) -> Dict[str, str]:
    run_logs = (
        Path(__file__).resolve().parent.parent
        / "econometrics"
        / "benchmarks"
        / "results"
        / "run_logs"
        / "compliance"
        / "claude-sonnet-4-20250514"
    )

    decisions: Dict[str, List[str]] = {cid: [] for cid in case_ids}

    for log_file in sorted(run_logs.glob("case_*_run_*.json")):
        if "_full" in log_file.name:
            continue
        with open(log_file) as f:
            data = json.load(f)
        cid = data.get("case_id", "")
        if cid in decisions:
            dec = data.get("decision_output", "").strip().lower()
            if dec:
                decisions[cid].append(dec)

    modal: Dict[str, str] = {}
    for cid, decs in decisions.items():
        if decs:
            modal[cid] = Counter(decs).most_common(1)[0][0]
    return modal


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    model: str,
    alerts_by_id: Dict[str, Dict],
    case_ids: List[str],
    group_label: str,
    n_runs: int = 3,
) -> List[Dict]:
    import anthropic as anth

    client = anth.Anthropic()
    results = []

    for case_id in case_ids:
        if case_id not in alerts_by_id:
            print(f"  SKIP {case_id}: not in test data")
            continue

        original = alerts_by_id[case_id]
        perturbed, pert_meta = perturb_alert(original)
        original_decision = ORIGINAL_DECISIONS.get(case_id, "unknown")

        print(
            f"  {case_id} (GT={original['ground_truth']}, "
            f"orig_modal={original_decision}, "
            f"${original['amount']:,.0f} -> ${perturbed['amount']:,.0f})"
        )

        run_decisions = []
        run_tools = []

        for r in range(n_runs):
            start = time.time()
            try:
                result = run_agent_anthropic(client, model, perturbed)
                latency = time.time() - start
                dec = result["decision"]
                tools = tuple(t["tool"] for t in result["tools_used"])
                run_decisions.append(dec)
                run_tools.append(tools)
                flipped = "FLIP" if dec != original_decision else "same"
                print(
                    f"    run {r}: {dec} ({flipped}) | "
                    f"tools={list(tools)} | {latency:.1f}s"
                )
            except Exception as e:
                print(f"    run {r}: ERROR — {e}")
                run_decisions.append("error")
                run_tools.append(())

            time.sleep(0.5)

        valid = [d for d in run_decisions if d != "error"]
        n_flipped = sum(1 for d in valid if d != original_decision)
        flip_rate = n_flipped / len(valid) if valid else None
        n_unique_tools = len(set(t for t in run_tools if t))

        results.append(
            {
                "case_id": case_id,
                "group": group_label,
                "ground_truth": original["ground_truth"],
                "original_modal_decision": original_decision,
                "original_amount": original["amount"],
                "perturbed_amount": perturbed["amount"],
                "perturbation_type": pert_meta["type"],
                "n_runs": n_runs,
                "perturbed_decisions": run_decisions,
                "perturbed_tool_sequences": [list(t) for t in run_tools],
                "n_flipped": n_flipped,
                "flip_rate": flip_rate,
                "n_unique_tool_sequences": n_unique_tools,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_results(results: List[Dict]) -> None:
    high = [r for r in results if r["group"] == "high_divergence"]
    low = [r for r in results if r["group"] == "low_divergence"]

    def group_stats(group, label):
        flip_rates = [r["flip_rate"] for r in group if r["flip_rate"] is not None]
        n_any = sum(1 for r in group if r["n_flipped"] > 0)
        mean_flip = sum(flip_rates) / len(flip_rates) if flip_rates else 0
        traj_var = [r["n_unique_tool_sequences"] for r in group]
        mean_traj_var = sum(traj_var) / len(traj_var) if traj_var else 0
        return {
            "label": label,
            "n_cases": len(group),
            "mean_flip_rate": mean_flip,
            "n_cases_any_flip": n_any,
            "pct_flipped": n_any / len(group) * 100 if group else 0,
            "flip_rates": flip_rates,
            "mean_traj_variation": mean_traj_var,
        }

    h = group_stats(high, "HIGH divergence (TAR < 0.9)")
    l = group_stats(low, "LOW divergence (TAR = 1.0)")

    print(f"\n{'=' * 72}")
    print("PERTURBATION EXPERIMENT RESULTS (v2: Compliance)")
    print(f"{'=' * 72}")
    print(f"Model: claude-sonnet-4-20250514")
    print(f"Task: Compliance Triage")
    print(f"Perturbation: transaction amount +15%")
    print()

    for s in [h, l]:
        print(f"  {s['label']}:")
        print(f"    Cases:                  {s['n_cases']}")
        print(f"    Mean decision flip rate: {s['mean_flip_rate']:.3f}")
        print(
            f"    Cases with any flip:     {s['n_cases_any_flip']} / "
            f"{s['n_cases']} ({s['pct_flipped']:.1f}%)"
        )
        print(f"    Mean traj variation:     {s['mean_traj_variation']:.2f} unique seqs/case")
        print(f"    Per-case flip rates:     {s['flip_rates']}")
        print()

    diff = h["mean_flip_rate"] - l["mean_flip_rate"]
    print(f"  Decision flip rate difference (HIGH - LOW): {diff:+.3f}")

    traj_diff = h["mean_traj_variation"] - l["mean_traj_variation"]
    print(f"  Trajectory variation diff  (HIGH - LOW): {traj_diff:+.2f}")
    print()

    if diff > 0.10:
        print("  STRONG SIGNAL: High-divergence cases are materially more fragile.")
        print("  DFAH trajectory divergence predicts decision instability under perturbation.")
    elif diff > 0.05:
        print("  MODERATE SIGNAL: High-divergence cases show more fragility.")
        print("  Effect exists but may need more cases to be robust.")
    elif diff > 0:
        print("  WEAK SIGNAL: Small positive difference.")
    else:
        print("  NO SIGNAL: High-divergence cases are not more fragile.")

    print(f"{'=' * 72}")

    # Per-case detail
    print(f"\nPer-case detail:")
    print(f"{'Case':<16} {'Group':<8} {'OrgDec':<12} {'Flips':>5} {'Rate':>6}  Perturbed decisions")
    print("-" * 80)
    for r in results:
        g = "HIGH" if r["group"] == "high_divergence" else "LOW"
        print(
            f"{r['case_id']:<16} {g:<8} {r['original_modal_decision']:<12} "
            f"{r['n_flipped']:>5} {r['flip_rate']:>6.2f}  {r['perturbed_decisions']}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global ORIGINAL_DECISIONS

    model = "claude-sonnet-4-20250514"
    n_runs = 3

    # Load test cases
    alerts = load_alerts()
    alerts_by_id = {a["alert_id"]: a for a in alerts}
    print(f"Loaded {len(alerts)} compliance alerts")

    # Load original decisions
    all_ids = HIGH_CASES + LOW_CASES
    ORIGINAL_DECISIONS = load_original_decisions(all_ids)
    print(f"Loaded original decisions for {len(ORIGINAL_DECISIONS)} cases")

    # Verify not pattern-matching
    dec_counts = Counter(ORIGINAL_DECISIONS.values())
    print(f"Original decision distribution: {dict(dec_counts)}")
    if len(dec_counts) < 2:
        print("WARNING: All cases have the same original decision — possible pattern matching!")

    for cid in all_ids:
        if cid not in ORIGINAL_DECISIONS:
            print(f"  WARNING: no corpus data for {cid}")

    # Run HIGH group
    print(f"\n{'=' * 72}")
    print(f"HIGH DIVERGENCE GROUP ({len(HIGH_CASES)} cases, {n_runs} runs each)")
    print(f"{'=' * 72}")
    high_results = run_experiment(
        model, alerts_by_id, HIGH_CASES, "high_divergence", n_runs
    )

    # Run LOW group
    print(f"\n{'=' * 72}")
    print(f"LOW DIVERGENCE GROUP ({len(LOW_CASES)} cases, {n_runs} runs each)")
    print(f"{'=' * 72}")
    low_results = run_experiment(
        model, alerts_by_id, LOW_CASES, "low_divergence", n_runs
    )

    all_results = high_results + low_results

    # Analyze
    analyze_results(all_results)

    # Save
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "experiment": "perturbation_validation_v2_compliance",
        "model": model,
        "benchmark": "compliance",
        "timestamp": datetime.now().isoformat(),
        "n_runs_per_case": n_runs,
        "perturbation": "amount_increase_15pct",
        "note": "v2: switched from the shared DataOps quarantine cue to rule-conditioned Compliance decisions",
        "results": all_results,
    }
    json_path = results_dir / "perturbation_experiment_v2.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    csv_path = results_dir / "perturbation_experiment_v2.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id", "group", "ground_truth", "original_modal_decision",
                "original_amount", "perturbed_amount", "n_runs",
                "n_flipped", "flip_rate", "n_unique_tool_sequences",
            ],
        )
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: v for k, v in r.items() if k in writer.fieldnames})

    print(f"\nResults saved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
