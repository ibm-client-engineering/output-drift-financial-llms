#!/usr/bin/env python3
"""
Run Compliance Triage Benchmark - Simple Version

Uses pre-provided evidence in the prompt to test decision determinism
without relying on tool calling (which has compatibility issues).
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import csv
import ollama


def create_evidence_prompt(alert: Dict) -> str:
    """Create a prompt with pre-retrieved evidence."""

    # Simulate tool results based on alert
    sanctions_hit = alert.get("receiver", "").lower() in ["shadow corp", "blocked entity"]
    risk_level = "high" if alert.get("amount", 0) > 100000 else "medium" if alert.get("amount", 0) > 10000 else "low"

    return f"""You are a compliance analyst. Based on the following alert and investigation results, decide:
- ESCALATE: High risk, needs full investigation
- DISMISS: False positive, normal business
- INVESTIGATE: Need more information

ALERT: {alert['alert_id']}
Amount: ${alert['amount']:,.2f} {alert['currency']}
Sender: {alert['sender']}
Receiver: {alert['receiver']}
Country: {alert['country']}
Flags: {', '.join(alert['flags'])}

INVESTIGATION RESULTS:
- Sanctions Check: {"MATCH FOUND - {alert['receiver']} appears on OFAC list" if sanctions_hit else "No match found"}
- Customer Profile: {alert['sender']} - {"New customer, no history" if "new_customer" in alert['flags'] else "Established customer, 5 year history"}
- Risk Score: {risk_level.upper()} risk ({0.85 if sanctions_hit else 0.3 if risk_level == 'medium' else 0.15})
- Precedent Search: {"Similar cases were ESCALATED" if sanctions_hit else "Similar cases were DISMISSED as normal business"}

Based on this evidence, what is your decision? Answer with exactly one word: ESCALATE, DISMISS, or INVESTIGATE"""


def load_alerts() -> List[Dict]:
    """Load test alerts."""
    alerts_path = Path(__file__).parent / "compliance_triage" / "data" / "alerts.json"

    if not alerts_path.exists():
        # Fallback to sample alerts
        return [
            {
                "alert_id": "TXN-SAMPLE-001",
                "amount": 47500.00,
                "currency": "USD",
                "sender": "ABC Corp",
                "receiver": "XYZ Holdings",
                "country": "Cayman Islands",
                "flags": ["unusual_amount", "offshore_destination"],
                "ground_truth": "dismiss"
            },
            {
                "alert_id": "TXN-SAMPLE-002",
                "amount": 125000.00,
                "currency": "USD",
                "sender": "New Customer LLC",
                "receiver": "Shadow Corp",
                "country": "Belarus",
                "flags": ["new_customer", "high_risk_country", "large_amount"],
                "ground_truth": "escalate"
            },
            {
                "alert_id": "TXN-SAMPLE-003",
                "amount": 5000.00,
                "currency": "USD",
                "sender": "Legitimate Inc",
                "receiver": "Supplier Co",
                "country": "Canada",
                "flags": ["round_amount"],
                "ground_truth": "dismiss"
            }
        ]

    with open(alerts_path) as f:
        data = json.load(f)
    return data["alerts"]


def run_experiment(model: str, alerts: List[Dict], num_runs: int = 8) -> Dict[str, Any]:
    """Run determinism experiment for a model."""

    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"Alerts: {len(alerts)}, Runs: {num_runs}")
    print(f"{'='*60}")

    client = ollama.Client()
    results = []

    for alert in alerts:
        prompt = create_evidence_prompt(alert)
        decisions = []

        print(f"\nAlert {alert['alert_id']} (GT: {alert['ground_truth']}):")

        for run in range(num_runs):
            start = time.time()
            try:
                resp = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.0, "seed": 42}
                )
                content = resp["message"]["content"].strip().upper()
                latency = time.time() - start

                # Parse decision
                if "ESCALATE" in content.split()[0] if content else False:
                    decision = "escalate"
                elif "DISMISS" in content.split()[0] if content else False:
                    decision = "dismiss"
                elif "INVESTIGATE" in content.split()[0] if content else False:
                    decision = "investigate"
                elif "ESCALATE" in content:
                    decision = "escalate"
                elif "DISMISS" in content:
                    decision = "dismiss"
                else:
                    decision = "investigate"

                decisions.append(decision)
                print(f"  Run {run+1}: {decision} ({latency:.1f}s) - '{content[:30]}...'")

            except Exception as e:
                decisions.append("error")
                print(f"  Run {run+1}: error - {e}")

        results.append({
            "alert_id": alert["alert_id"],
            "ground_truth": alert["ground_truth"],
            "decisions": decisions,
            "is_deterministic": len(set(decisions)) == 1,
            "is_correct": decisions[0] == alert["ground_truth"] if decisions else False
        })

    # Calculate metrics
    total_alerts = len(results)
    deterministic = sum(1 for r in results if r["is_deterministic"])
    correct = sum(1 for r in results if r["is_correct"])

    return {
        "model": model,
        "num_alerts": total_alerts,
        "num_runs": num_runs,
        "decision_determinism": 100 * deterministic / total_alerts,
        "accuracy": 100 * correct / total_alerts,
        "details": results,
        "timestamp": datetime.now().isoformat()
    }


def main():
    """Run experiments on Tier 1 models."""

    models = [
        "qwen2.5:7b-instruct",
        "granite3.3:latest",
        "gpt-oss:20b",
    ]

    alerts = load_alerts()[:15]  # Test on 15 alerts
    all_results = []

    for model in models:
        try:
            result = run_experiment(model, alerts, num_runs=8)
            all_results.append(result)

            print(f"\n{model} Results:")
            print(f"  Decision Determinism: {result['decision_determinism']:.1f}%")
            print(f"  Accuracy: {result['accuracy']:.1f}%")

        except Exception as e:
            print(f"Error with {model}: {e}")

    # Save results
    output_path = Path(__file__).parent.parent.parent / "results" / "v3_compliance_simple.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "="*60)
    print("COMPLIANCE TRIAGE - ACTUAL EXPERIMENTAL RESULTS")
    print("="*60)
    print(f"{'Model':<25} {'Det%':>8} {'Acc%':>8}")
    print("-"*45)
    for r in all_results:
        print(f"{r['model']:<25} {r['decision_determinism']:>8.1f} {r['accuracy']:>8.1f}")


if __name__ == "__main__":
    main()
