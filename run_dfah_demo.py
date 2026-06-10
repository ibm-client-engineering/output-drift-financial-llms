#!/usr/bin/env python3
"""
Determinism-Faithfulness Assurance Harness (DFAH) - Quick Demo

Runs financial agent benchmarks to measure trajectory determinism and accuracy.
From the paper: "Replayable Financial Agents" (arXiv:2601.15322, ICLR 2026 FinAI Workshop)

No LLM or API keys required - uses deterministic agent simulation.

Usage:
    python run_dfah_demo.py                    # Quick demo (5 cases, 3 runs)
    python run_dfah_demo.py --full             # Full benchmark (50 cases, 8 runs)
    python run_dfah_demo.py --task compliance  # Single task

Output:
    JSON results saved to dfah_results/
    See DFAH.md for output schema and customization guide.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Ensure econometrics package is importable
sys.path.insert(0, str(Path(__file__).parent))

from econometrics.benchmarks.run_all import run_benchmark, run_all_benchmarks
from dataclasses import asdict


OUTPUT_DIR = Path(__file__).parent / "dfah_results"


def save_results(results, n_cases, n_runs):
    """Save DFAH results to JSON."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    output = {
        "dfah_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "config": {"n_cases": n_cases, "n_runs_per_case": n_runs},
        "benchmarks": {}
    }

    for name, metrics in results.items():
        output["benchmarks"][name] = {
            "action_determinism": round(metrics.action_determinism, 1),
            "signature_determinism": round(metrics.signature_determinism, 1),
            "decision_determinism": round(metrics.decision_determinism, 1),
            "accuracy": round(metrics.accuracy, 1),
            "n_tests": metrics.n_tests,
            "n_runs_per_test": metrics.n_runs_per_test,
            "ground_truth_distribution": metrics.ground_truth_distribution,
        }

    results_file = OUTPUT_DIR / "dfah_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    return results_file


def print_header():
    print()
    print("=" * 65)
    print("  DFAH: Determinism-Faithfulness Assurance Harness")
    print("  arXiv:2601.15322 | ICLR 2026 FinAI Workshop")
    print("=" * 65)
    print()
    print("  Measures three independent dimensions of agent behavior:")
    print("    1. Signature Determinism - identical tool call sequences?")
    print("    2. Decision Determinism  - identical final decisions?")
    print("    3. Accuracy              - decisions match ground truth?")
    print()
    print("  Key finding: determinism and accuracy are NOT correlated")
    print("  (r = -0.11, p = 0.63 across 4,705 agentic runs)")
    print()


def print_footer(results_file):
    print()
    print("-" * 65)
    print("  WHAT THESE RESULTS MEAN")
    print("-" * 65)
    print()
    print("  This simulation uses fixed logic (no LLM), so all metrics")
    print("  are 100%. With real LLMs, you see behavioral profiles:")
    print()
    print("    Pattern Matchers (Qwen 7B):   98% determ. / 33% accuracy")
    print("    Balanced (Claude Sonnet):      84% determ. / 38% accuracy")
    print("    Explorers (Claude Opus):       71% determ. / 44% accuracy")
    print()
    print(f"  Results saved to: {results_file}")
    print()
    print("  Next steps:")
    print("    - See DFAH.md for output schema and customization")
    print("    - See examples/dfah_custom_task.py to bring your own cases")
    print("    - Run with a real LLM (requires Ollama):")
    print("        python econometrics/benchmarks/run_agentic_benchmark.py \\")
    print("          --model qwen2.5:7b-instruct --n-cases 5 --n-runs 4")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="DFAH - Determinism-Faithfulness Assurance Harness"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full benchmark (50 cases, 8 runs per case)"
    )
    parser.add_argument(
        "--task",
        choices=["compliance", "portfolio", "dataops", "all"],
        default="all",
        help="Which benchmark task to run (default: all)"
    )
    parser.add_argument(
        "--n-cases", type=int, default=None,
        help="Number of test cases (default: 5 for demo, 50 for --full)"
    )
    parser.add_argument(
        "--n-runs", type=int, default=None,
        help="Number of runs per case (default: 3 for demo, 8 for --full)"
    )
    args = parser.parse_args()

    # Set defaults based on mode
    if args.full:
        n_cases = args.n_cases or 50
        n_runs = args.n_runs or 8
    else:
        n_cases = args.n_cases or 5
        n_runs = args.n_runs or 3

    print_header()

    task_map = {
        "compliance": "compliance_triage",
        "portfolio": "portfolio_constraint",
        "dataops": "dataops_exception",
        "all": "all",
    }
    task = task_map[args.task]

    print(f"  Mode: {'Full benchmark' if args.full else 'Quick demo'}")
    print(f"  Cases: {n_cases} | Runs per case: {n_runs}")
    print(f"  Task: {task}")
    print()

    if task == "all":
        results = run_all_benchmarks(n_runs=n_runs, max_tests=n_cases)
    else:
        metrics = run_benchmark(task, n_runs=n_runs, max_tests=n_cases)
        results = {task: metrics} if metrics else {}

    # Save structured output
    results_file = save_results(results, n_cases, n_runs)

    print_footer(results_file)


if __name__ == "__main__":
    main()
