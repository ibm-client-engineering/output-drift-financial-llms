#!/usr/bin/env python3
"""
Run Compliance Triage Benchmark Experiments

Executes the compliance triage task against Ollama models to generate
real experimental data for ICLR 2026 paper Table 4.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import csv

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import ollama

from compliance_triage.task import (
    ComplianceTriageTools,
    TransactionAlert,
    TriageDecision,
    MockToolContext,
    create_test_context,
    SAMPLE_ALERTS,
    GROUND_TRUTH
)


@dataclass
class ExperimentConfig:
    """Configuration for a benchmark run."""
    model: str
    temperature: float = 0.0
    seed: int = 42
    num_runs: int = 8
    timeout: int = 120


@dataclass
class RunResult:
    """Result from a single agent run."""
    alert_id: str
    run_id: int
    decision: str
    tools_called: List[str]
    tool_signatures: List[Dict]
    reasoning: str
    latency_s: float
    raw_response: str


def load_full_alerts() -> List[TransactionAlert]:
    """Load all 50 test alerts from JSON file."""
    alerts_path = Path(__file__).parent / "compliance_triage" / "data" / "alerts.json"

    if not alerts_path.exists():
        print(f"Warning: {alerts_path} not found, using sample alerts")
        return SAMPLE_ALERTS

    with open(alerts_path) as f:
        data = json.load(f)

    alerts = []
    for item in data["alerts"]:
        alerts.append(TransactionAlert(
            alert_id=item["alert_id"],
            amount=item["amount"],
            currency=item["currency"],
            sender=item["sender"],
            receiver=item["receiver"],
            country=item["country"],
            flags=item["flags"],
            description=item.get("description", "")
        ))

    return alerts


def load_ground_truth() -> Dict[str, TriageDecision]:
    """Load ground truth from JSON file."""
    alerts_path = Path(__file__).parent / "compliance_triage" / "data" / "alerts.json"

    if not alerts_path.exists():
        return GROUND_TRUTH

    with open(alerts_path) as f:
        data = json.load(f)

    gt = {}
    for item in data["alerts"]:
        decision_str = item["ground_truth"]
        gt[item["alert_id"]] = TriageDecision(decision_str)

    return gt


def create_system_prompt() -> str:
    """Create the system prompt for compliance triage."""
    return """You are a compliance analyst assistant. Your job is to triage transaction alerts and decide whether to:

1. ESCALATE - Forward to compliance team for full investigation (high risk)
2. DISMISS - Close as false positive (low risk, normal business)
3. INVESTIGATE - Request additional information before deciding

You have access to these tools:
- search_precedents(query): Search historical compliance cases
- get_customer_profile(customer_id): Get customer risk profile
- check_sanctions_list(name): Screen against OFAC sanctions
- calculate_risk_score(factors): Compute risk score

IMPORTANT:
- Use tools to gather evidence before making a decision
- Base your decision ONLY on evidence from tools
- Respond with your final decision in the format: DECISION: [ESCALATE|DISMISS|INVESTIGATE]
- Provide brief reasoning citing the evidence"""


def run_single_alert(
    client: ollama.Client,
    config: ExperimentConfig,
    alert: TransactionAlert,
    tools: ComplianceTriageTools,
    run_id: int
) -> RunResult:
    """Run the agent on a single alert."""

    tools.call_log = []  # Reset tool call log

    prompt = f"""{create_system_prompt()}

{alert.to_prompt()}"""

    start_time = time.time()

    try:
        # Call Ollama with tool definitions
        response = client.chat(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            tools=tools.get_tools_schema(),
            options={
                "temperature": config.temperature,
                "seed": config.seed,
            }
        )

        # Process tool calls if present
        message = response.get("message", {})
        tool_calls = message.get("tool_calls", [])

        # Execute any tool calls
        for tc in tool_calls:
            func_name = tc.get("function", {}).get("name", "")
            args = tc.get("function", {}).get("arguments", {})

            if hasattr(tools, func_name):
                func = getattr(tools, func_name)
                if isinstance(args, dict):
                    func(**args)
                elif isinstance(args, str):
                    # Try to parse as JSON
                    try:
                        parsed = json.loads(args)
                        func(**parsed)
                    except:
                        pass

        raw_response = message.get("content", "")

        # Extract decision from response
        decision = "INVESTIGATE"  # default
        content_upper = raw_response.upper()
        if "DECISION: ESCALATE" in content_upper or "ESCALATE" in content_upper.split()[-5:]:
            decision = "escalate"
        elif "DECISION: DISMISS" in content_upper or "DISMISS" in content_upper.split()[-5:]:
            decision = "dismiss"
        elif "DECISION: INVESTIGATE" in content_upper:
            decision = "investigate"

        latency = time.time() - start_time

        return RunResult(
            alert_id=alert.alert_id,
            run_id=run_id,
            decision=decision,
            tools_called=[c["tool"] for c in tools.call_log],
            tool_signatures=tools.call_log.copy(),
            reasoning=raw_response[:500],
            latency_s=latency,
            raw_response=raw_response
        )

    except Exception as e:
        latency = time.time() - start_time
        return RunResult(
            alert_id=alert.alert_id,
            run_id=run_id,
            decision="error",
            tools_called=[],
            tool_signatures=[],
            reasoning=f"Error: {str(e)}",
            latency_s=latency,
            raw_response=""
        )


def calculate_determinism(results: List[RunResult]) -> Dict[str, float]:
    """Calculate determinism metrics from run results."""

    # Group by alert_id
    by_alert: Dict[str, List[RunResult]] = {}
    for r in results:
        if r.alert_id not in by_alert:
            by_alert[r.alert_id] = []
        by_alert[r.alert_id].append(r)

    decision_matches = 0
    action_matches = 0
    signature_matches = 0
    total = 0

    for alert_id, runs in by_alert.items():
        if len(runs) < 2:
            continue

        first = runs[0]
        for run in runs[1:]:
            total += 1

            # Decision determinism
            if run.decision == first.decision:
                decision_matches += 1

            # Action determinism (same tools, any order)
            if set(run.tools_called) == set(first.tools_called):
                action_matches += 1

            # Signature determinism (same tools with same args)
            if run.tool_signatures == first.tool_signatures:
                signature_matches += 1

    if total == 0:
        return {"decision_det": 0, "action_det": 0, "signature_det": 0}

    return {
        "decision_det": 100 * decision_matches / total,
        "action_det": 100 * action_matches / total,
        "signature_det": 100 * signature_matches / total
    }


def calculate_accuracy(results: List[RunResult], ground_truth: Dict[str, TriageDecision]) -> float:
    """Calculate accuracy against ground truth."""
    correct = 0
    total = 0

    for r in results:
        if r.alert_id in ground_truth and r.decision != "error":
            total += 1
            if r.decision == ground_truth[r.alert_id].value:
                correct += 1

    return 100 * correct / total if total > 0 else 0


def run_experiment(config: ExperimentConfig, max_alerts: int = 10) -> Dict[str, Any]:
    """Run full experiment for a model configuration."""

    print(f"\n{'='*60}")
    print(f"Running experiment: {config.model}")
    print(f"Temperature: {config.temperature}, Runs: {config.num_runs}")
    print(f"{'='*60}")

    # Load alerts and ground truth
    all_alerts = load_full_alerts()
    alerts = all_alerts[:max_alerts]  # Limit for faster testing
    ground_truth = load_ground_truth()

    print(f"Loaded {len(alerts)} alerts (of {len(all_alerts)} total)")

    # Initialize Ollama client and tools
    client = ollama.Client()
    context = create_test_context()
    tools = ComplianceTriageTools(context)

    # Run experiments
    all_results: List[RunResult] = []

    for alert in alerts:
        print(f"\nAlert {alert.alert_id}:")
        for run_id in range(config.num_runs):
            result = run_single_alert(client, config, alert, tools, run_id)
            all_results.append(result)
            print(f"  Run {run_id+1}: {result.decision} ({result.latency_s:.2f}s)")

    # Calculate metrics
    determinism = calculate_determinism(all_results)
    accuracy = calculate_accuracy(all_results, ground_truth)

    avg_latency = sum(r.latency_s for r in all_results) / len(all_results)

    return {
        "model": config.model,
        "temperature": config.temperature,
        "num_alerts": len(alerts),
        "num_runs": config.num_runs,
        "total_calls": len(all_results),
        "decision_determinism": determinism["decision_det"],
        "action_determinism": determinism["action_det"],
        "signature_determinism": determinism["signature_det"],
        "accuracy": accuracy,
        "mean_latency_s": avg_latency,
        "timestamp": datetime.now().isoformat()
    }


def main():
    """Run benchmark experiments for paper Table 4."""

    # Models to test (Tier 1 candidates)
    models = [
        "qwen2.5:7b-instruct",
        "deepseek-r1:8b",
        "granite3.3:latest",
    ]

    configs = [
        ExperimentConfig(model=m, temperature=0.0, num_runs=8)
        for m in models
    ]

    results = []

    for config in configs:
        try:
            result = run_experiment(config, max_alerts=10)
            results.append(result)
            print(f"\nResults for {config.model}:")
            print(f"  Decision Det: {result['decision_determinism']:.1f}%")
            print(f"  Action Det: {result['action_determinism']:.1f}%")
            print(f"  Accuracy: {result['accuracy']:.1f}%")
        except Exception as e:
            print(f"Error running {config.model}: {e}")

    # Save results
    output_path = Path(__file__).parent.parent.parent / "results" / "v3_compliance_benchmark.csv"
    output_path.parent.mkdir(exist_ok=True)

    if results:
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_path}")

    # Print summary table for paper
    print("\n" + "="*60)
    print("PAPER TABLE 4 DATA - Compliance Triage Benchmark")
    print("="*60)
    print(f"{'Model':<25} {'Dec.Det%':>10} {'Act.Det%':>10} {'Acc%':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['model']:<25} {r['decision_determinism']:>10.1f} {r['action_determinism']:>10.1f} {r['accuracy']:>10.1f}")


if __name__ == "__main__":
    main()
