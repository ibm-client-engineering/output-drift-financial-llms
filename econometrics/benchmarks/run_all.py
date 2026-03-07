#!/usr/bin/env python3
"""
Unified Benchmark Runner for V3 Financial Agent Evaluation

Runs all three benchmark tasks and computes determinism/faithfulness metrics.
Supports multiple model configurations and produces results for paper tables.

Usage:
    python econometrics/benchmarks/run_all.py [--task TASK] [--n-runs N]

Output:
    - Console summary of metrics
    - JSON results in benchmarks/results/
    - CSV for paper tables
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from econometrics.benchmarks.compliance_triage.task import (
    ComplianceTriageTools, create_test_context as create_compliance_context,
    TransactionAlert, GROUND_TRUTH as COMPLIANCE_GROUND_TRUTH
)
from econometrics.benchmarks.portfolio_constraint.task import (
    PortfolioConstraintTools, create_test_context as create_portfolio_context,
    ProposedTrade, GROUND_TRUTH as PORTFOLIO_GROUND_TRUTH
)
from econometrics.benchmarks.dataops_exception.task import (
    DataOpsTools, create_test_context as create_dataops_context,
    DataException, ExceptionType, GROUND_TRUTH as DATAOPS_GROUND_TRUTH
)

# Paths
BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    benchmark: str
    test_id: str
    run_id: int
    tools_called: List[str]
    tool_signatures: List[Dict]
    decision: str
    ground_truth: str
    correct: bool
    timestamp: str


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a benchmark."""
    benchmark: str
    n_tests: int
    n_runs_per_test: int
    action_determinism: float
    signature_determinism: float
    decision_determinism: float
    accuracy: float
    ground_truth_distribution: Dict[str, int]


def load_benchmark_data(benchmark: str) -> List[Dict]:
    """Load test data for a benchmark."""
    data_files = {
        "compliance_triage": BENCHMARK_DIR / "compliance_triage/data/alerts.json",
        "portfolio_constraint": BENCHMARK_DIR / "portfolio_constraint/data/trades.json",
        "dataops_exception": BENCHMARK_DIR / "dataops_exception/data/exceptions.json"
    }

    data_path = data_files.get(benchmark)
    if not data_path or not data_path.exists():
        print(f"Warning: No data file for {benchmark}, using sample data")
        return []

    with open(data_path) as f:
        data = json.load(f)

    if benchmark == "compliance_triage":
        return data.get("alerts", [])
    elif benchmark == "portfolio_constraint":
        return data.get("trades", [])
    elif benchmark == "dataops_exception":
        return data.get("exceptions", [])

    return []


def simulate_compliance_agent(alert: Dict, tools: ComplianceTriageTools) -> Dict:
    """Simulate an agent processing a compliance alert."""
    tools.call_log = []

    # Simulate typical agent behavior
    # 1. Check sanctions
    sanctions = tools.check_sanctions_list(alert["receiver"])

    # 2. Get customer profile
    profile = tools.get_customer_profile(alert["sender"])

    # 3. Search precedents
    precedents = tools.search_precedents(" ".join(alert.get("flags", [])))

    # 4. Calculate risk
    risk = tools.calculate_risk_score({
        "amount": alert["amount"],
        "offshore": alert["country"] not in ["USA", "Canada", "UK", "Germany", "France"],
        "new_counterparty": profile.get("relationship_years", 0) == 0,
        "sanctions_hit": sanctions["is_sanctioned"]
    })

    # Determine decision based on risk score
    if sanctions["is_sanctioned"] or risk["risk_score"] > 0.7:
        decision = "escalate"
    elif risk["risk_score"] < 0.3 and not sanctions["is_sanctioned"]:
        decision = "dismiss"
    else:
        decision = "investigate"

    return {
        "tools_called": [c["tool"] for c in tools.call_log],
        "tool_signatures": tools.call_log.copy(),
        "decision": decision
    }


def simulate_portfolio_agent(trade: Dict, tools: PortfolioConstraintTools) -> Dict:
    """Simulate an agent processing a portfolio trade."""
    tools.call_log = []

    # 1. Get holdings
    holdings = tools.get_current_holdings(trade.get("portfolio_id", "FUND-2025-ALPHA"))

    # 2. Get market data
    market = tools.get_market_data(trade["ticker"])

    # 3. Check position limit
    notional = trade["quantity"] * trade["price"]
    pos_check = tools.check_position_limit(
        trade["ticker"],
        trade["quantity"],
        holdings["total_value"]
    )

    # 4. Check sector exposure
    sector = tools.calculate_sector_exposure(
        market.get("sector", "Unknown"),
        trade.get("portfolio_id", "FUND-2025-ALPHA")
    )

    # Determine decision
    if not pos_check["within_limit"]:
        decision = "reject"
    elif trade["action"] == "sell":
        decision = "approve"
    else:
        # Check liquidity
        volume = market.get("volume_3d_avg", 1000000)
        days_to_trade = notional / (volume * market.get("price", 100))
        if days_to_trade > 3:
            decision = "reject"
        else:
            decision = "approve"

    return {
        "tools_called": [c["tool"] for c in tools.call_log],
        "tool_signatures": tools.call_log.copy(),
        "decision": decision
    }


def simulate_dataops_agent(exception: Dict, tools: DataOpsTools) -> Dict:
    """Simulate an agent processing a data exception."""
    tools.call_log = []

    # 1. Get exception details
    details = tools.get_exception_details(exception["exception_id"])

    # 2. Search historical fixes
    hist = tools.get_historical_fixes(exception["rule_violated"])

    # 3. Determine action based on exception type
    exc_type = exception.get("exception_type", "")

    if exc_type == "format_error":
        # Try to validate a fix
        if exception["field"] in ["trade_date", "record_date"]:
            # Date format conversion
            tools.validate_fix(exception["field"], exception["value"], "2025-01-15")
            decision = "auto_fix"
        elif exception["field"] in ["trade_price", "quantity", "volume"]:
            # Numeric fixes
            try:
                new_val = abs(float(exception["value"])) if exception["value"] else 0
                tools.validate_fix(exception["field"], exception["value"], new_val)
                decision = "auto_fix"
            except (ValueError, TypeError):
                decision = "escalate"
        else:
            decision = "auto_fix"

    elif exc_type == "business_rule":
        # Check if simple fix (negative -> positive)
        if isinstance(exception["value"], (int, float)) and exception["value"] < 0:
            new_val = abs(exception["value"])
            validation = tools.validate_fix(exception["field"], exception["value"], new_val)
            decision = "auto_fix" if validation["is_valid"] else "escalate"
        elif exception["value"] == 0:
            decision = "escalate"  # Zero values need review
        else:
            decision = "quarantine"  # Out of bounds

    elif exc_type == "reference_mismatch":
        # Try reference lookup
        ref = tools.query_reference_data(exception["field"], str(exception["value"]))
        if ref["match_found"]:
            decision = "auto_fix"
        else:
            decision = "escalate"

    elif exc_type == "missing_field":
        # Missing fields usually need escalation
        decision = "escalate"

    else:
        decision = "quarantine"

    return {
        "tools_called": [c["tool"] for c in tools.call_log],
        "tool_signatures": tools.call_log.copy(),
        "decision": decision
    }


def run_benchmark(benchmark: str, n_runs: int = 5, max_tests: int = 50) -> BenchmarkMetrics:
    """Run a benchmark and compute metrics."""
    print(f"\n{'='*60}")
    print(f"Running benchmark: {benchmark}")
    print(f"{'='*60}")

    # Load test data
    tests = load_benchmark_data(benchmark)
    if not tests:
        print(f"No test data found for {benchmark}")
        return None

    tests = tests[:max_tests]
    print(f"Loaded {len(tests)} test cases")

    # Create tools context
    if benchmark == "compliance_triage":
        context = create_compliance_context()
        tools = ComplianceTriageTools(context)
        simulate_fn = simulate_compliance_agent
    elif benchmark == "portfolio_constraint":
        context = create_portfolio_context()
        tools = PortfolioConstraintTools(context)
        simulate_fn = simulate_portfolio_agent
    elif benchmark == "dataops_exception":
        context = create_dataops_context()
        tools = DataOpsTools(context)
        simulate_fn = simulate_dataops_agent
    else:
        print(f"Unknown benchmark: {benchmark}")
        return None

    # Run tests
    results: List[BenchmarkResult] = []
    test_results: Dict[str, List[Dict]] = {}

    for test in tests:
        test_id = test.get("alert_id") or test.get("trade_id") or test.get("exception_id")
        ground_truth = test.get("ground_truth", "unknown")
        test_results[test_id] = []

        for run_id in range(n_runs):
            result = simulate_fn(test, tools)

            results.append(BenchmarkResult(
                benchmark=benchmark,
                test_id=test_id,
                run_id=run_id,
                tools_called=result["tools_called"],
                tool_signatures=result["tool_signatures"],
                decision=result["decision"],
                ground_truth=ground_truth,
                correct=result["decision"] == ground_truth,
                timestamp=datetime.now().isoformat()
            ))

            test_results[test_id].append(result)

    # Compute metrics
    action_matches = 0
    signature_matches = 0
    decision_matches = 0
    total_comparisons = 0

    for test_id, runs in test_results.items():
        if len(runs) < 2:
            continue

        first_run = runs[0]
        for other_run in runs[1:]:
            total_comparisons += 1

            # Action determinism: same tools called
            if set(first_run["tools_called"]) == set(other_run["tools_called"]):
                action_matches += 1

            # Signature determinism: same tools with same args
            if first_run["tool_signatures"] == other_run["tool_signatures"]:
                signature_matches += 1

            # Decision determinism: same final decision
            if first_run["decision"] == other_run["decision"]:
                decision_matches += 1

    # Compute accuracy
    correct = sum(1 for r in results if r.correct)
    total = len(results)

    metrics = BenchmarkMetrics(
        benchmark=benchmark,
        n_tests=len(tests),
        n_runs_per_test=n_runs,
        action_determinism=action_matches / total_comparisons * 100 if total_comparisons > 0 else 0,
        signature_determinism=signature_matches / total_comparisons * 100 if total_comparisons > 0 else 0,
        decision_determinism=decision_matches / total_comparisons * 100 if total_comparisons > 0 else 0,
        accuracy=correct / total * 100 if total > 0 else 0,
        ground_truth_distribution={gt: sum(1 for t in tests if t.get("ground_truth") == gt)
                                   for gt in set(t.get("ground_truth") for t in tests)}
    )

    # Print results
    print(f"\n{benchmark.upper()} RESULTS:")
    print(f"  Tests: {metrics.n_tests}")
    print(f"  Runs per test: {metrics.n_runs_per_test}")
    print(f"  Action Determinism: {metrics.action_determinism:.1f}%")
    print(f"  Signature Determinism: {metrics.signature_determinism:.1f}%")
    print(f"  Decision Determinism: {metrics.decision_determinism:.1f}%")
    print(f"  Accuracy: {metrics.accuracy:.1f}%")
    print(f"  Ground Truth Distribution: {metrics.ground_truth_distribution}")

    return metrics


def run_all_benchmarks(n_runs: int = 5, max_tests: int = 50) -> Dict[str, BenchmarkMetrics]:
    """Run all benchmarks and return combined results."""
    benchmarks = ["compliance_triage", "portfolio_constraint", "dataops_exception"]
    results = {}

    for benchmark in benchmarks:
        metrics = run_benchmark(benchmark, n_runs, max_tests)
        if metrics:
            results[benchmark] = metrics

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = RESULTS_DIR / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, 'w') as f:
        json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_file}")

    # Print summary table
    print(f"\n{'Benchmark':<25} {'Action Det.':<12} {'Sig. Det.':<12} {'Dec. Det.':<12} {'Accuracy':<10}")
    print("-" * 75)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics.action_determinism:>10.1f}% {metrics.signature_determinism:>10.1f}% "
              f"{metrics.decision_determinism:>10.1f}% {metrics.accuracy:>8.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run V3 Financial Agent Benchmarks")
    parser.add_argument("--task", type=str, choices=["compliance_triage", "portfolio_constraint", "dataops_exception", "all"],
                        default="all", help="Which benchmark to run")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of runs per test case")
    parser.add_argument("--max-tests", type=int, default=50, help="Maximum test cases to run")

    args = parser.parse_args()

    print("=" * 60)
    print("V3 FINANCIAL AGENT BENCHMARK SUITE")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: {args.n_runs} runs/test, max {args.max_tests} tests")

    if args.task == "all":
        results = run_all_benchmarks(args.n_runs, args.max_tests)
    else:
        results = {args.task: run_benchmark(args.task, args.n_runs, args.max_tests)}

    return results


if __name__ == "__main__":
    main()
