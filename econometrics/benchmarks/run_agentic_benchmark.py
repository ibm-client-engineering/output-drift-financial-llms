#!/usr/bin/env python3
"""
Agentic Compliance Triage Benchmark

Proper tool-calling benchmark for ICLR 2026 paper.
Tests actual agentic behavior with tool use and multi-turn conversations.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Global stress mode
STRESS_MODE = "baseline"  # baseline, dq_fault, vol_shock

try:
    import ollama
except ImportError:
    ollama = None

try:
    import anthropic
except ImportError:
    anthropic = None

# TODO: Add Gemini support in future
# try:
#     import google.generativeai as genai
# except ImportError:
#     genai = None


# Tool definitions in proper Ollama format
COMPLIANCE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_sanctions",
            "description": "Check if an entity name appears on OFAC sanctions list",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "The entity name to screen"
                    }
                },
                "required": ["entity_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_profile",
            "description": "Get risk profile and KYC status for a customer",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer name or ID"
                    }
                },
                "required": ["customer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_risk_score",
            "description": "Calculate overall risk score for a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Transaction amount"},
                    "is_offshore": {"type": "boolean", "description": "Is destination offshore"},
                    "is_new_customer": {"type": "boolean", "description": "Is customer new"},
                    "sanctions_hit": {"type": "boolean", "description": "Any sanctions matches"}
                },
                "required": ["amount"]
            }
        }
    }
]

# Anthropic tool format
COMPLIANCE_TOOLS_ANTHROPIC = [
    {
        "name": "check_sanctions",
        "description": "Check if an entity name appears on OFAC sanctions list",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_name": {"type": "string", "description": "The entity name to screen"}
            },
            "required": ["entity_name"]
        }
    },
    {
        "name": "get_customer_profile",
        "description": "Get risk profile and KYC status for a customer",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "Customer name or ID"}
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "calculate_risk_score",
        "description": "Calculate overall risk score for a transaction",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Transaction amount"},
                "is_offshore": {"type": "boolean", "description": "Is destination offshore"},
                "is_new_customer": {"type": "boolean", "description": "Is customer new"},
                "sanctions_hit": {"type": "boolean", "description": "Any sanctions matches"}
            },
            "required": ["amount"]
        }
    }
]

# TODO: Add COMPLIANCE_TOOLS_GEMINI when Gemini support is needed

# Simulated tool responses
SANCTIONS_DB = {
    "shadow corp": True,
    "blocked entity": True,
    "suspicious ltd": True,
}

CUSTOMER_DB = {
    "new customer llc": {"risk_level": "high", "kyc_status": "pending", "years": 0},
    "abc corp": {"risk_level": "low", "kyc_status": "complete", "years": 5},
    "legitimate inc": {"risk_level": "low", "kyc_status": "complete", "years": 3},
}


def inject_dq_fault(result: Dict) -> Dict:
    """Inject data quality faults: 10% chance of NULL/missing values."""
    if random.random() < 0.10:  # 10% fault rate
        # Pick a random key to corrupt
        keys = list(result.keys())
        if keys:
            corrupt_key = random.choice(keys)
            result[corrupt_key] = None  # Inject NULL
    return result


def inject_vol_shock(result: Dict, name: str) -> Dict:
    """Inject volatility shock: ±3σ on numerical values."""
    if name == "calculate_risk_score" and "risk_score" in result:
        # Add ±3σ noise (σ ≈ 0.15 for risk scores)
        shock = random.gauss(0, 0.15) * 3
        original = result["risk_score"]
        result["risk_score"] = max(0.0, min(1.0, original + shock))
        # Recalculate level based on shocked score
        score = result["risk_score"]
        result["risk_level"] = "HIGH" if score > 0.6 else "MEDIUM" if score > 0.3 else "LOW"
    return result


def execute_tool(name: str, args: Dict) -> Dict:
    """Execute a tool and return result, with optional stress injection."""
    global STRESS_MODE

    if name == "check_sanctions":
        entity = args.get("entity_name", "").lower()
        is_hit = SANCTIONS_DB.get(entity, False)
        result = {
            "entity": args.get("entity_name"),
            "is_sanctioned": is_hit,
            "list": "OFAC SDN" if is_hit else None,
            "match_score": 1.0 if is_hit else 0.0
        }

    elif name == "get_customer_profile":
        cust = args.get("customer_id", "").lower()
        profile = CUSTOMER_DB.get(cust, {
            "risk_level": "unknown",
            "kyc_status": "incomplete",
            "years": 0
        })
        result = {
            "customer": args.get("customer_id"),
            "risk_level": profile["risk_level"],
            "kyc_status": profile["kyc_status"],
            "relationship_years": profile["years"]
        }

    elif name == "calculate_risk_score":
        score = 0.0
        if args.get("amount", 0) > 50000:
            score += 0.3
        if args.get("is_offshore", False):
            score += 0.2
        if args.get("is_new_customer", False):
            score += 0.2
        if args.get("sanctions_hit", False):
            score += 0.4
        result = {
            "risk_score": min(score, 1.0),
            "risk_level": "HIGH" if score > 0.6 else "MEDIUM" if score > 0.3 else "LOW"
        }

    else:
        return {"error": f"Unknown tool: {name}"}

    # Apply stress injection
    if STRESS_MODE == "dq_fault":
        result = inject_dq_fault(result)
    elif STRESS_MODE == "vol_shock":
        result = inject_vol_shock(result, name)

    return result


def run_agent(client: ollama.Client, model: str, alert: Dict, max_turns: int = 5) -> Dict:
    """Run the agent on an alert with multi-turn tool calling."""

    system_prompt = """You are a compliance analyst. Analyze the alert and decide:
- ESCALATE: Forward to compliance team (high risk indicators)
- DISMISS: Close as false positive (normal business)
- INVESTIGATE: Need more information

IMPORTANT: Use the tools to gather evidence BEFORE deciding.
After gathering evidence, state your final decision clearly as: DECISION: [ESCALATE/DISMISS/INVESTIGATE]"""

    user_prompt = f"""COMPLIANCE ALERT: {alert['alert_id']}

Transaction:
- Amount: ${alert['amount']:,.2f} {alert['currency']}
- Sender: {alert['sender']}
- Receiver: {alert['receiver']}
- Destination: {alert['country']}
- Flags: {', '.join(alert['flags'])}

Use the available tools to investigate, then provide your decision."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    tools_used = []
    final_decision = None

    for turn in range(max_turns):
        resp = client.chat(
            model=model,
            messages=messages,
            tools=COMPLIANCE_TOOLS,
            options={"temperature": 0.0, "seed": 42}
        )

        msg = resp.get("message", {})
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # Check for final decision in content
        if content:
            content_upper = content.upper()
            if "DECISION:" in content_upper:
                if "ESCALATE" in content_upper:
                    final_decision = "escalate"
                elif "DISMISS" in content_upper:
                    final_decision = "dismiss"
                elif "INVESTIGATE" in content_upper:
                    final_decision = "investigate"

            # Also check last words
            if not final_decision:
                words = content_upper.split()[-10:] if content else []
                if "ESCALATE" in words:
                    final_decision = "escalate"
                elif "DISMISS" in words:
                    final_decision = "dismiss"

        # Process tool calls
        if tool_calls:
            messages.append(msg)

            for tc in tool_calls:
                # Handle ToolCall object from ollama
                try:
                    if hasattr(tc, 'function'):
                        func = tc.function
                        name = getattr(func, 'name', '') or ''
                        args = getattr(func, 'arguments', {}) or {}
                    else:
                        name = tc.get("function", {}).get("name", "")
                        args = tc.get("function", {}).get("arguments", {})
                except Exception as e:
                    print(f"    Tool parse error: {e}")
                    name = ""
                    args = {}

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}

                if name:
                    result = execute_tool(name, args)
                    tools_used.append({"tool": name, "args": args, "result": result})

                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result)
                    })

        # If we have a decision and no more tool calls, we're done
        if final_decision and not tool_calls:
            break

        # If no tool calls and no decision, model might be done
        if not tool_calls:
            break

    # If still no decision, default based on content
    if not final_decision:
        if content:
            cu = content.upper()
            if "ESCALATE" in cu:
                final_decision = "escalate"
            elif "DISMISS" in cu:
                final_decision = "dismiss"
            else:
                final_decision = "investigate"
        else:
            final_decision = "investigate"

    return {
        "decision": final_decision,
        "tools_used": tools_used,
        "num_turns": turn + 1,
        "final_content": content[:500] if content else ""
    }


def run_agent_anthropic(client, model: str, alert: Dict, max_turns: int = 5) -> Dict:
    """Run the agent on an alert using Anthropic API."""

    system_prompt = """You are a compliance analyst. Analyze the alert and decide:
- ESCALATE: Forward to compliance team (high risk indicators)
- DISMISS: Close as false positive (normal business)
- INVESTIGATE: Need more information

IMPORTANT: Use the tools to gather evidence BEFORE deciding.
After gathering evidence, state your final decision clearly as: DECISION: [ESCALATE/DISMISS/INVESTIGATE]"""

    user_prompt = f"""COMPLIANCE ALERT: {alert['alert_id']}

Transaction:
- Amount: ${alert['amount']:,.2f} {alert['currency']}
- Sender: {alert['sender']}
- Receiver: {alert['receiver']}
- Destination: {alert['country']}
- Flags: {', '.join(alert['flags'])}

Use the available tools to investigate, then provide your decision."""

    messages = [{"role": "user", "content": user_prompt}]

    tools_used = []
    final_decision = None
    content = ""

    for turn in range(max_turns):
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.0,  # replay protocol (audit fix: was provider default 1.0)
            system=system_prompt,
            tools=COMPLIANCE_TOOLS_ANTHROPIC,
            messages=messages
        )

        # Process response
        assistant_content = []
        tool_use_blocks = []

        for block in resp.content:
            if block.type == "text":
                content = block.text
                assistant_content.append({"type": "text", "text": content})

                # Check for decision
                content_upper = content.upper()
                if "DECISION:" in content_upper:
                    if "ESCALATE" in content_upper:
                        final_decision = "escalate"
                    elif "DISMISS" in content_upper:
                        final_decision = "dismiss"
                    elif "INVESTIGATE" in content_upper:
                        final_decision = "investigate"

            elif block.type == "tool_use":
                tool_use_blocks.append(block)
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })

        # Add assistant message
        messages.append({"role": "assistant", "content": assistant_content})

        # Process tool calls
        if tool_use_blocks:
            tool_results = []
            for block in tool_use_blocks:
                result = execute_tool(block.name, block.input)
                tools_used.append({"tool": block.name, "args": block.input, "result": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result)
                })

            messages.append({"role": "user", "content": tool_results})

        # Check stop condition
        if resp.stop_reason == "end_turn" and not tool_use_blocks:
            break

        if final_decision and not tool_use_blocks:
            break

    # Default decision
    if not final_decision:
        if content:
            cu = content.upper()
            if "ESCALATE" in cu:
                final_decision = "escalate"
            elif "DISMISS" in cu:
                final_decision = "dismiss"
            else:
                final_decision = "investigate"
        else:
            final_decision = "investigate"

    return {
        "decision": final_decision,
        "tools_used": tools_used,
        "num_turns": turn + 1,
        "final_content": content[:500] if content else ""
    }


def load_alerts() -> List[Dict]:
    """Load test alerts."""
    alerts_path = Path(__file__).parent / "compliance_triage" / "data" / "alerts.json"

    if alerts_path.exists():
        with open(alerts_path) as f:
            data = json.load(f)
        return data["alerts"]

    # Fallback samples
    return [
        {
            "alert_id": "TXN-TEST-001",
            "amount": 47500.00,
            "currency": "USD",
            "sender": "ABC Corp",
            "receiver": "XYZ Holdings",
            "country": "Cayman Islands",
            "flags": ["offshore_destination"],
            "ground_truth": "dismiss"
        },
        {
            "alert_id": "TXN-TEST-002",
            "amount": 125000.00,
            "currency": "USD",
            "sender": "New Customer LLC",
            "receiver": "Shadow Corp",
            "country": "Belarus",
            "flags": ["new_customer", "high_risk_country", "large_amount"],
            "ground_truth": "escalate"
        },
        {
            "alert_id": "TXN-TEST-003",
            "amount": 5000.00,
            "currency": "USD",
            "sender": "Legitimate Inc",
            "receiver": "Supplier Co",
            "country": "Canada",
            "flags": ["round_amount"],
            "ground_truth": "dismiss"
        }
    ]


def run_experiment(model: str, alerts: List[Dict], num_runs: int = 8, provider: str = "ollama") -> Dict:
    """Run full experiment."""

    print(f"\n{'='*60}")
    print(f"AGENTIC BENCHMARK: {model} ({provider})")
    print(f"Alerts: {len(alerts)}, Runs per alert: {num_runs}")
    print(f"{'='*60}")

    if provider == "ollama":
        if ollama is None:
            raise ImportError("ollama package not installed")
        client = ollama.Client()
        agent_fn = lambda alert: run_agent(client, model, alert)
    elif provider == "anthropic":
        if anthropic is None:
            raise ImportError("anthropic package not installed")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        client = anthropic.Anthropic(api_key=api_key)
        agent_fn = lambda alert: run_agent_anthropic(client, model, alert)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    all_results = []

    for alert in alerts:
        print(f"\nAlert {alert['alert_id']} (GT: {alert['ground_truth']}):")

        decisions = []
        tool_counts = []

        for run in range(num_runs):
            start = time.time()
            result = agent_fn(alert)
            latency = time.time() - start

            decisions.append(result["decision"])
            tool_counts.append(len(result["tools_used"]))

            tool_names = [t["tool"] for t in result["tools_used"]]
            print(f"  Run {run+1}: {result['decision']} | Tools: {tool_names} | {latency:.1f}s")

        all_results.append({
            "alert_id": alert["alert_id"],
            "ground_truth": alert["ground_truth"],
            "decisions": decisions,
            "tool_counts": tool_counts,
            "is_deterministic": len(set(decisions)) == 1,
            "is_correct": decisions[0] == alert["ground_truth"],
            "avg_tools": sum(tool_counts) / len(tool_counts)
        })

    # Calculate metrics
    n = len(all_results)
    det = sum(1 for r in all_results if r["is_deterministic"])
    correct = sum(1 for r in all_results if r["is_correct"])
    avg_tools = sum(r["avg_tools"] for r in all_results) / n

    return {
        "model": model,
        "num_alerts": n,
        "num_runs": num_runs,
        "decision_determinism": 100 * det / n,
        "accuracy": 100 * correct / n,
        "avg_tools_per_run": avg_tools,
        "details": all_results,
        "timestamp": datetime.now().isoformat()
    }


def main():
    """Run agentic benchmark."""
    global STRESS_MODE

    parser = argparse.ArgumentParser(description="Agentic Compliance Triage Benchmark")
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct",
                        help="Model to run (default: qwen2.5:7b-instruct)")
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "anthropic"],
                        help="Provider to use (default: ollama)")
    parser.add_argument("--n-cases", type=int, default=10,
                        help="Number of alert cases to test (default: 10)")
    parser.add_argument("--n-runs", type=int, default=8,
                        help="Number of runs per case (default: 8)")
    parser.add_argument("--stress", type=str, default="baseline",
                        choices=["baseline", "dq_fault", "vol_shock"],
                        help="Stress test mode (default: baseline)")
    parser.add_argument("--all-models", action="store_true",
                        help="Run all preconfigured models")
    args = parser.parse_args()

    # Set global stress mode
    STRESS_MODE = args.stress
    if STRESS_MODE != "baseline":
        print(f"\n*** STRESS MODE: {STRESS_MODE} ***")

    alerts = load_alerts()[:args.n_cases]

    if args.all_models:
        # Models with tool calling support (validated)
        models = [
            "qwen2.5:7b-instruct",
            "gpt-oss:20b",
        ]
    else:
        models = [args.model]

    results = []
    for model in models:
        try:
            r = run_experiment(model, alerts, num_runs=args.n_runs, provider=args.provider)
            results.append(r)
            print(f"\n{model}:")
            print(f"  Decision Determinism: {r['decision_determinism']:.1f}%")
            print(f"  Accuracy: {r['accuracy']:.1f}%")
            print(f"  Avg Tools/Run: {r['avg_tools_per_run']:.1f}")
        except Exception as e:
            print(f"Error with {model}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Use model name in output file
    model_slug = args.model.replace(":", "_").replace("/", "_")
    output = output_dir / f"v3_agentic_{model_slug}.json"
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output}")

    # Summary
    print("\n" + "="*60)
    print("AGENTIC BENCHMARK RESULTS (for Paper Table 4)")
    print("="*60)
    print(f"{'Model':<25} {'Det%':>8} {'Acc%':>8} {'Tools':>8}")
    print("-"*50)
    for r in results:
        print(f"{r['model']:<25} {r['decision_determinism']:>8.1f} {r['accuracy']:>8.1f} {r['avg_tools_per_run']:>8.1f}")


if __name__ == "__main__":
    main()
