#!/usr/bin/env python3
"""
DataOps Exception Benchmark Runner

Evaluates LLM agent determinism and accuracy on data quality exception
resolution tasks. Tests tool-calling behavior across multiple runs.

Usage:
    # Ollama (local)
    python run_dataops_benchmark.py --model qwen2.5:7b-instruct --n-cases 10 --n-runs 8

    # Anthropic API
    python run_dataops_benchmark.py --model claude-opus-4-5 --provider anthropic --n-cases 10 --n-runs 8
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import ollama
except ImportError:
    ollama = None

try:
    import anthropic
except ImportError:
    anthropic = None


# Tool definitions in Ollama format
DATAOPS_TOOLS_OLLAMA = [
    {
        "type": "function",
        "function": {
            "name": "get_exception_details",
            "description": "Get full context about a data exception including priority, SLA deadline, and similar exceptions",
            "parameters": {
                "type": "object",
                "properties": {
                    "exception_id": {
                        "type": "string",
                        "description": "The exception ID to look up"
                    }
                },
                "required": ["exception_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_reference_data",
            "description": "Look up canonical value in reference data (tickers, CUSIPs, currencies)",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": "The field type to query (e.g., ticker, cusip, currency)"
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to look up"
                    }
                },
                "required": ["field", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_fixes",
            "description": "Search past fixes for similar exception patterns",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to search for (e.g., negative_price, date_format)"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_fix",
            "description": "Validate a proposed fix before applying it",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": "The field being fixed"
                    },
                    "old_value": {
                        "type": "string",
                        "description": "The current invalid value"
                    },
                    "new_value": {
                        "type": "string",
                        "description": "The proposed new value"
                    }
                },
                "required": ["field", "old_value", "new_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_fix",
            "description": "Apply and log a fix to the exception",
            "parameters": {
                "type": "object",
                "properties": {
                    "exception_id": {
                        "type": "string",
                        "description": "The exception ID to fix"
                    },
                    "fix": {
                        "type": "object",
                        "description": "The fix details including action and new_value"
                    }
                },
                "required": ["exception_id", "fix"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate exception to human review when automatic resolution is not possible",
            "parameters": {
                "type": "object",
                "properties": {
                    "exception_id": {
                        "type": "string",
                        "description": "The exception ID to escalate"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for escalation"
                    }
                },
                "required": ["exception_id", "reason"]
            }
        }
    }
]

# Anthropic tool format
DATAOPS_TOOLS_ANTHROPIC = [
    {
        "name": "get_exception_details",
        "description": "Get full context about a data exception including priority, SLA deadline, and similar exceptions",
        "input_schema": {
            "type": "object",
            "properties": {
                "exception_id": {
                    "type": "string",
                    "description": "The exception ID to look up"
                }
            },
            "required": ["exception_id"]
        }
    },
    {
        "name": "query_reference_data",
        "description": "Look up canonical value in reference data (tickers, CUSIPs, currencies)",
        "input_schema": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "description": "The field type to query (e.g., ticker, cusip, currency)"
                },
                "value": {
                    "type": "string",
                    "description": "The value to look up"
                }
            },
            "required": ["field", "value"]
        }
    },
    {
        "name": "get_historical_fixes",
        "description": "Search past fixes for similar exception patterns",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for (e.g., negative_price, date_format)"
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "validate_fix",
        "description": "Validate a proposed fix before applying it",
        "input_schema": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "description": "The field being fixed"
                },
                "old_value": {
                    "type": "string",
                    "description": "The current invalid value"
                },
                "new_value": {
                    "type": "string",
                    "description": "The proposed new value"
                }
            },
            "required": ["field", "old_value", "new_value"]
        }
    },
    {
        "name": "apply_fix",
        "description": "Apply and log a fix to the exception",
        "input_schema": {
            "type": "object",
            "properties": {
                "exception_id": {
                    "type": "string",
                    "description": "The exception ID to fix"
                },
                "fix": {
                    "type": "object",
                    "description": "The fix details including action and new_value"
                }
            },
            "required": ["exception_id", "fix"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate exception to human review when automatic resolution is not possible",
        "input_schema": {
            "type": "object",
            "properties": {
                "exception_id": {
                    "type": "string",
                    "description": "The exception ID to escalate"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for escalation"
                }
            },
            "required": ["exception_id", "reason"]
        }
    }
]


# Mock data context for tool responses
REFERENCE_DATA = {
    "ticker": {
        "MSFT": {"canonical": "MSFT", "alternatives": ["MICROSOFT", "MSFT.O"]},
        "AAPL": {"canonical": "AAPL", "alternatives": ["APPLE", "AAPL.O"]},
        "GOOG": {"canonical": "GOOGL", "alternatives": ["GOOGLE", "GOOG.O"]},
        "GOOGL": {"canonical": "GOOGL", "alternatives": ["GOOGLE", "GOOG"]},
    },
    "cusip": {
        "594918104": {"canonical": "594918104", "name": "MSFT"},
        "037833100": {"canonical": "037833100", "name": "AAPL"},
        "02079K107": {"canonical": "02079K107", "name": "GOOGL"},
    },
    "currency": {
        "DOLLAR": {"canonical": "USD", "alternatives": ["US DOLLAR", "USDOLLAR"]},
        "EURO": {"canonical": "EUR", "alternatives": ["EUROS"]},
        "USD": {"canonical": "USD"},
        "EUR": {"canonical": "EUR"},
    }
}

HISTORICAL_FIXES = {
    "negative": [
        {"pattern": "negative_value", "resolution": "absolute_value", "success_rate": 0.95, "count": 150}
    ],
    "price": [
        {"pattern": "negative_price", "resolution": "absolute_value", "success_rate": 0.95, "count": 150},
        {"pattern": "zero_price", "resolution": "escalate", "success_rate": 0.0, "count": 30}
    ],
    "date": [
        {"pattern": "MM/DD/YYYY", "resolution": "convert_to_ISO", "success_rate": 1.0, "count": 500},
        {"pattern": "invalid_date", "resolution": "escalate", "success_rate": 0.0, "count": 25}
    ],
    "format": [
        {"pattern": "format_conversion", "resolution": "convert_format", "success_rate": 0.98, "count": 300}
    ],
    "ticker": [
        {"pattern": "alternative_ticker", "resolution": "map_to_canonical", "success_rate": 0.98, "count": 200}
    ],
    "missing": [
        {"pattern": "missing_cusip", "resolution": "escalate", "success_rate": 0.0, "count": 50},
        {"pattern": "missing_required", "resolution": "escalate", "success_rate": 0.0, "count": 100}
    ],
    "bounds": [
        {"pattern": "out_of_bounds", "resolution": "quarantine", "success_rate": 0.0, "count": 45}
    ]
}


def execute_tool(name: str, args: Dict) -> Dict:
    """Execute a tool and return simulated result."""
    if name == "get_exception_details":
        exc_id = args.get("exception_id", "UNKNOWN")
        return {
            "exception_id": exc_id,
            "created_at": "2025-01-15T10:30:00Z",
            "source_system": "market_data_feed",
            "priority": "high",
            "sla_deadline": "2025-01-15T11:30:00Z",
            "similar_exceptions_today": 3
        }

    elif name == "query_reference_data":
        field = args.get("field", "").lower()
        value = args.get("value", "")
        ref = REFERENCE_DATA.get(field, {})
        match = ref.get(value.upper() if isinstance(value, str) else str(value))
        return {
            "field": field,
            "query_value": value,
            "canonical_value": match.get("canonical") if match else None,
            "match_found": match is not None,
            "alternatives": match.get("alternatives", []) if match else []
        }

    elif name == "get_historical_fixes":
        pattern = args.get("pattern", "").lower()
        results = []
        for key, fixes in HISTORICAL_FIXES.items():
            if key in pattern or pattern in key:
                results.extend(fixes)
        return {"pattern": pattern, "fixes": results[:5], "total_found": len(results)}

    elif name == "validate_fix":
        field = args.get("field", "")
        old_value = args.get("old_value")
        new_value = args.get("new_value")
        is_valid = True
        errors = []

        # Basic validation
        if "price" in field.lower() or "amount" in field.lower():
            try:
                nv = float(new_value) if isinstance(new_value, str) else new_value
                if nv is not None and nv < 0:
                    is_valid = False
                    errors.append("Value cannot be negative")
            except (ValueError, TypeError):
                pass

        return {
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
            "is_valid": is_valid,
            "validation_errors": errors
        }

    elif name == "apply_fix":
        exc_id = args.get("exception_id", "UNKNOWN")
        fix = args.get("fix", {})
        return {
            "exception_id": exc_id,
            "status": "fixed",
            "fix_applied": fix,
            "audit_log_id": f"AUDIT-{exc_id}"
        }

    elif name == "escalate_to_human":
        exc_id = args.get("exception_id", "UNKNOWN")
        reason = args.get("reason", "No reason provided")
        return {
            "exception_id": exc_id,
            "status": "escalated",
            "escalation_queue": "data_quality_team",
            "reason": reason,
            "estimated_response": "1 hour"
        }

    return {"error": f"Unknown tool: {name}"}


def run_agent_ollama(client, model: str, exception: Dict, max_turns: int = 6) -> Dict:
    """Run the agent on an exception using Ollama."""

    system_prompt = """You are a data quality analyst handling exceptions in a financial data pipeline.
Analyze the exception and decide:
- AUTO_FIX: Apply automatic correction when the fix is clear and safe
- ESCALATE: Requires human review when the issue is ambiguous or high-risk
- QUARANTINE: Cannot determine action, data needs investigation

IMPORTANT: Use the tools to gather information and validate fixes BEFORE deciding.
After analysis, state your final decision clearly as: DECISION: [AUTO_FIX/ESCALATE/QUARANTINE]"""

    user_prompt = f"""DATA QUALITY EXCEPTION: {exception['exception_id']}

Source: {exception['source']}
Exception Type: {exception['exception_type']}
Field: {exception['field']}
Invalid Value: {exception['value']}
Rule Violated: {exception['rule_violated']}

Full Record:
{json.dumps(exception['record'], indent=2)}

Use the available tools to research the issue and determine the best action."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    tools_used = []
    final_decision = None
    content = ""

    for turn in range(max_turns):
        resp = client.chat(
            model=model,
            messages=messages,
            tools=DATAOPS_TOOLS_OLLAMA,
            options={"temperature": 0.0, "seed": 42}
        )

        msg = resp.get("message", {})
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # Check for final decision in content
        if content:
            content_upper = content.upper()
            if "DECISION:" in content_upper or any(d in content_upper for d in ["AUTO_FIX", "ESCALATE", "QUARANTINE"]):
                if "AUTO_FIX" in content_upper or "AUTO-FIX" in content_upper or "AUTOFIX" in content_upper:
                    final_decision = "auto_fix"
                elif "ESCALATE" in content_upper:
                    final_decision = "escalate"
                elif "QUARANTINE" in content_upper:
                    final_decision = "quarantine"

        # Process tool calls
        if tool_calls:
            messages.append(msg)

            for tc in tool_calls:
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

        # If no tool calls, model might be done
        if not tool_calls:
            break

    # Default decision based on content if not explicitly stated
    if not final_decision:
        if content:
            cu = content.upper()
            if "AUTO_FIX" in cu or "FIX" in cu or "CORRECT" in cu:
                final_decision = "auto_fix"
            elif "ESCALATE" in cu or "HUMAN" in cu or "REVIEW" in cu:
                final_decision = "escalate"
            elif "QUARANTINE" in cu or "INVESTIGATE" in cu:
                final_decision = "quarantine"
            else:
                final_decision = "escalate"  # Default to safe option
        else:
            final_decision = "escalate"

    return {
        "decision": final_decision,
        "tools_used": tools_used,
        "num_turns": turn + 1,
        "final_content": content[:500] if content else ""
    }


def run_agent_anthropic(client, model: str, exception: Dict, max_turns: int = 6) -> Dict:
    """Run the agent on an exception using Anthropic API."""

    system_prompt = """You are a data quality analyst handling exceptions in a financial data pipeline.
Analyze the exception and decide:
- AUTO_FIX: Apply automatic correction when the fix is clear and safe
- ESCALATE: Requires human review when the issue is ambiguous or high-risk
- QUARANTINE: Cannot determine action, data needs investigation

IMPORTANT: Use the tools to gather information and validate fixes BEFORE deciding.
After analysis, state your final decision clearly as: DECISION: [AUTO_FIX/ESCALATE/QUARANTINE]"""

    user_prompt = f"""DATA QUALITY EXCEPTION: {exception['exception_id']}

Source: {exception['source']}
Exception Type: {exception['exception_type']}
Field: {exception['field']}
Invalid Value: {exception['value']}
Rule Violated: {exception['rule_violated']}

Full Record:
{json.dumps(exception['record'], indent=2)}

Use the available tools to research the issue and determine the best action."""

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
            tools=DATAOPS_TOOLS_ANTHROPIC,
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
                if "DECISION:" in content_upper or any(d in content_upper for d in ["AUTO_FIX", "ESCALATE", "QUARANTINE"]):
                    if "AUTO_FIX" in content_upper or "AUTO-FIX" in content_upper:
                        final_decision = "auto_fix"
                    elif "ESCALATE" in content_upper:
                        final_decision = "escalate"
                    elif "QUARANTINE" in content_upper:
                        final_decision = "quarantine"

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
            if "AUTO_FIX" in cu or "FIX" in cu:
                final_decision = "auto_fix"
            elif "ESCALATE" in cu or "HUMAN" in cu:
                final_decision = "escalate"
            else:
                final_decision = "escalate"
        else:
            final_decision = "escalate"

    return {
        "decision": final_decision,
        "tools_used": tools_used,
        "num_turns": turn + 1,
        "final_content": content[:500] if content else ""
    }


def load_exceptions(path: Optional[Path] = None) -> List[Dict]:
    """Load exceptions from JSON file."""
    if path is None:
        path = Path(__file__).parent / "dataops_exception" / "data" / "exceptions.json"

    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data.get("exceptions", [])

    print(f"Warning: {path} not found, using sample exceptions")
    return []


def run_experiment(
    model: str,
    provider: str,
    exceptions: List[Dict],
    num_runs: int = 8
) -> Dict:
    """Run full experiment."""

    print(f"\n{'='*60}")
    print(f"DATAOPS BENCHMARK: {model} ({provider})")
    print(f"Exceptions: {len(exceptions)}, Runs per exception: {num_runs}")
    print(f"{'='*60}")

    if provider == "ollama":
        if ollama is None:
            raise ImportError("ollama package not installed")
        client = ollama.Client()
        run_agent = lambda exc: run_agent_ollama(client, model, exc)
    elif provider == "anthropic":
        if anthropic is None:
            raise ImportError("anthropic package not installed")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        client = anthropic.Anthropic(api_key=api_key)
        run_agent = lambda exc: run_agent_anthropic(client, model, exc)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    all_results = []

    for exception in exceptions:
        print(f"\nException {exception['exception_id']} (GT: {exception['ground_truth']}):")

        decisions = []
        tool_sequences = []
        tool_counts = []

        for run in range(num_runs):
            start = time.time()
            try:
                result = run_agent(exception)
                latency = time.time() - start

                decisions.append(result["decision"])
                tool_names = [t["tool"] for t in result["tools_used"]]
                tool_sequences.append(tuple(tool_names))
                tool_counts.append(len(result["tools_used"]))

                print(f"  Run {run+1}: {result['decision']} | Tools: {tool_names} | {latency:.1f}s")
            except Exception as e:
                print(f"  Run {run+1}: ERROR - {e}")
                decisions.append("error")
                tool_sequences.append(())
                tool_counts.append(0)

        # Calculate metrics
        valid_decisions = [d for d in decisions if d != "error"]
        is_deterministic = len(set(valid_decisions)) == 1 if valid_decisions else False
        is_correct = valid_decisions[0] == exception["ground_truth"] if valid_decisions else False

        # Signature determinism: same tools in same order
        valid_sequences = [s for s in tool_sequences if s]
        signature_deterministic = len(set(valid_sequences)) == 1 if valid_sequences else False

        # Action determinism: same tools regardless of order
        tool_sets = [frozenset(s) for s in valid_sequences if s]
        action_deterministic = len(set(tool_sets)) == 1 if tool_sets else False

        all_results.append({
            "exception_id": exception["exception_id"],
            "exception_type": exception["exception_type"],
            "ground_truth": exception["ground_truth"],
            "decisions": decisions,
            "tool_counts": tool_counts,
            "decision_deterministic": is_deterministic,
            "signature_deterministic": signature_deterministic,
            "action_deterministic": action_deterministic,
            "is_correct": is_correct,
            "avg_tools": sum(tool_counts) / len(tool_counts) if tool_counts else 0
        })

    # Aggregate metrics
    n = len(all_results)
    if n == 0:
        return {"error": "No results"}

    decision_det = sum(1 for r in all_results if r["decision_deterministic"])
    signature_det = sum(1 for r in all_results if r["signature_deterministic"])
    action_det = sum(1 for r in all_results if r["action_deterministic"])
    correct = sum(1 for r in all_results if r["is_correct"])
    avg_tools = sum(r["avg_tools"] for r in all_results) / n

    return {
        "model": model,
        "provider": provider,
        "benchmark": "DataOps Exception",
        "num_exceptions": n,
        "num_runs": num_runs,
        "metrics": {
            "decision_determinism": 100 * decision_det / n,
            "signature_determinism": 100 * signature_det / n,
            "action_determinism": 100 * action_det / n,
            "accuracy": 100 * correct / n,
            "avg_tools_per_run": avg_tools
        },
        "details": all_results,
        "timestamp": datetime.now().isoformat()
    }


def main():
    parser = argparse.ArgumentParser(description="DataOps Exception Benchmark")
    parser.add_argument("--model", default="qwen2.5:7b-instruct", help="Model to use")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "anthropic"])
    parser.add_argument("--n-cases", type=int, default=10, help="Number of exceptions to test")
    parser.add_argument("--n-runs", type=int, default=8, help="Runs per exception")
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()

    # Load exceptions
    exceptions = load_exceptions()[:args.n_cases]
    if not exceptions:
        print("No exceptions loaded!")
        return

    print(f"Loaded {len(exceptions)} exceptions")

    # Run experiment
    try:
        results = run_experiment(
            model=args.model,
            provider=args.provider,
            exceptions=exceptions,
            num_runs=args.n_runs
        )
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print summary
    print("\n" + "=" * 60)
    print("DATAOPS BENCHMARK RESULTS")
    print("=" * 60)
    m = results["metrics"]
    print(f"Model: {results['model']} ({results['provider']})")
    print(f"Exceptions: {results['num_exceptions']}, Runs: {results['num_runs']}")
    print("-" * 40)
    print(f"Decision Determinism:  {m['decision_determinism']:.1f}%")
    print(f"Signature Determinism: {m['signature_determinism']:.1f}%")
    print(f"Action Determinism:    {m['action_determinism']:.1f}%")
    print(f"Accuracy:              {m['accuracy']:.1f}%")
    print(f"Avg Tools/Run:         {m['avg_tools_per_run']:.1f}")

    # Save results
    output_path = args.output or Path(__file__).parent.parent.parent / "results" / "v3_dataops_benchmark.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results and append
    existing = []
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            existing = [existing]

    existing.append(results)

    with open(output_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
