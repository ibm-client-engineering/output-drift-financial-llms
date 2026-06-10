#!/usr/bin/env python3
"""Audit divergence channel coverage across existing replay logs.

Scans replay trace JSON files and reports a matrix by benchmark/model
showing availability of each divergence channel:
  - decision outputs
  - tool/trajectory data
  - evidence contacts (derived from tool outputs)
  - reasoning text

This determines which DFAH-Bench metrics are computable per model/benchmark
and prevents overclaiming in the paper.

Usage:
    python scripts/audit_channel_coverage.py
    python scripts/audit_channel_coverage.py --run-logs-dir path/to/run_logs
    python scripts/audit_channel_coverage.py --output audit_results.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def find_run_logs_dir() -> Path:
    """Locate the run_logs directory relative to this script."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    candidates = [
        repo_root / "econometrics" / "benchmarks" / "results" / "run_logs",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    print("ERROR: Could not find run_logs directory.", file=sys.stderr)
    print("Tried:", file=sys.stderr)
    for c in candidates:
        print(f"  {c}", file=sys.stderr)
    sys.exit(1)


def assess_episode(data: Dict[str, Any]) -> Dict[str, bool]:
    """Assess which divergence channels are available in a single episode."""
    channels = {
        "decision": False,
        "trajectory": False,
        "evidence_contacts": False,
        "reasoning_text": False,
    }

    # Decision: present if decision_output is a non-empty string
    decision = data.get("decision_output", "")
    if isinstance(decision, str) and decision.strip():
        channels["decision"] = True

    # Trajectory: present if tool_sequence is a non-empty list
    tool_seq = data.get("tool_sequence", [])
    if isinstance(tool_seq, list) and len(tool_seq) > 0:
        channels["trajectory"] = True

    # Evidence contacts: present if tool_outputs has content
    # (from _full.json logs with actual tool return values)
    tool_outputs = data.get("tool_outputs", [])
    if isinstance(tool_outputs, list) and len(tool_outputs) > 0:
        # Check that at least one output has meaningful content
        for output in tool_outputs:
            if output and output != {}:
                channels["evidence_contacts"] = True
                break

    # Also check tool_output_hashes as a weaker signal
    if not channels["evidence_contacts"]:
        hashes = data.get("tool_output_hashes", [])
        if isinstance(hashes, list) and len(hashes) > 0:
            channels["evidence_contacts"] = True

    # Reasoning text: present if there's a reasoning/rationale/content field
    for field in ["reasoning_text", "rationale", "final_content", "raw_response"]:
        value = data.get(field, "")
        if isinstance(value, str) and len(value) > 20:
            channels["reasoning_text"] = True
            break

    return channels


def scan_run_logs(run_logs_dir: Path) -> Dict[Tuple[str, str], Dict[str, Dict[str, int]]]:
    """Scan all run logs and aggregate channel availability by benchmark/model.

    Returns:
        Dict mapping (benchmark, model) to channel counts:
        {
            ("compliance", "mistral_7b"): {
                "total_episodes": 400,
                "decision": 400,
                "trajectory": 12,
                "evidence_contacts": 0,
                "reasoning_text": 0,
            }
        }
    """
    results: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {
            "total_episodes": 0,
            "decision": 0,
            "trajectory": 0,
            "evidence_contacts": 0,
            "reasoning_text": 0,
        }
    )

    if not run_logs_dir.is_dir():
        print(f"ERROR: {run_logs_dir} is not a directory", file=sys.stderr)
        return {}

    # Walk benchmark/model/case_*.json
    for benchmark_dir in sorted(run_logs_dir.iterdir()):
        if not benchmark_dir.is_dir():
            continue
        benchmark = benchmark_dir.name

        for model_dir in sorted(benchmark_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name

            key = (benchmark, model)

            for log_file in sorted(model_dir.glob("case_*_run_*.json")):
                # Skip _full.json files in the count — they're supplementary
                # But DO scan them for channel availability
                if "_full" in log_file.name:
                    continue

                try:
                    with open(log_file) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

                results[key]["total_episodes"] += 1
                channels = assess_episode(data)

                for channel, available in channels.items():
                    if available:
                        results[key][channel] += 1

                # Also check corresponding _full.json for richer data
                full_path = log_file.with_name(
                    log_file.name.replace(".json", "_full.json")
                )
                if full_path.exists():
                    try:
                        with open(full_path) as f:
                            full_data = json.load(f)
                        full_channels = assess_episode(full_data)
                        # Update with richer data from full log
                        for channel in ["evidence_contacts", "reasoning_text"]:
                            if full_channels[channel] and not channels[channel]:
                                results[key][channel] += 1
                    except (json.JSONDecodeError, OSError):
                        pass

    return dict(results)


def format_coverage_symbol(count: int, total: int) -> str:
    """Format coverage as a symbol for the matrix."""
    if total == 0:
        return "-"
    ratio = count / total
    if ratio == 0:
        return "none"
    elif ratio < 0.1:
        return f"rare ({count}/{total})"
    elif ratio < 0.9:
        return f"partial ({count}/{total})"
    elif ratio < 1.0:
        return f"most ({count}/{total})"
    else:
        return "all"


def print_matrix(results: Dict[Tuple[str, str], Dict[str, int]]) -> None:
    """Print the channel-availability matrix."""
    if not results:
        print("No run logs found.")
        return

    # Header
    print("\n=== DFAH-Bench Channel Coverage Audit ===\n")
    print(f"{'Benchmark':<15} {'Model':<30} {'Episodes':>8}  "
          f"{'Decisions':<12} {'Trajectories':<16} {'Evidence':<16} {'Reasoning':<16}")
    print("-" * 115)

    for (benchmark, model), counts in sorted(results.items()):
        total = counts["total_episodes"]
        print(
            f"{benchmark:<15} {model:<30} {total:>8}  "
            f"{format_coverage_symbol(counts['decision'], total):<12} "
            f"{format_coverage_symbol(counts['trajectory'], total):<16} "
            f"{format_coverage_symbol(counts['evidence_contacts'], total):<16} "
            f"{format_coverage_symbol(counts['reasoning_text'], total):<16}"
        )

    # Summary
    total_episodes = sum(c["total_episodes"] for c in results.values())
    total_with_traj = sum(c["trajectory"] for c in results.values())
    total_with_evidence = sum(c["evidence_contacts"] for c in results.values())
    total_with_reasoning = sum(c["reasoning_text"] for c in results.values())

    print("-" * 115)
    print(f"\nTotal: {total_episodes} episodes across {len(results)} benchmark-model configs")
    print(f"  Decisions:     {sum(c['decision'] for c in results.values())}/{total_episodes}")
    print(f"  Trajectories:  {total_with_traj}/{total_episodes}")
    print(f"  Evidence:      {total_with_evidence}/{total_episodes}")
    print(f"  Reasoning:     {total_with_reasoning}/{total_episodes}")

    # Metric supportability
    print("\n=== Metric Supportability ===\n")
    print("DCB:  Computable for ALL configs (only needs decisions)")
    if total_with_traj > 0:
        print(f"SCDR (trajectory mode): Computable for {sum(1 for c in results.values() if c['trajectory'] > 0)} configs")
    else:
        print("SCDR (trajectory mode): NOT computable (no trajectory data)")
    if total_with_evidence > 0:
        print(f"ECD:  Computable for {sum(1 for c in results.values() if c['evidence_contacts'] > 0)} configs")
    else:
        print("ECD:  NOT computable (no evidence contact data)")
    if total_with_reasoning > 0:
        print(f"SCDR (rationale mode): Computable for {sum(1 for c in results.values() if c['reasoning_text'] > 0)} configs")
    else:
        print("SCDR (rationale mode): NOT computable (no reasoning text)")


def export_json(results: Dict[Tuple[str, str], Dict[str, int]], output_path: Path) -> None:
    """Export results as JSON for programmatic use."""
    serializable = {}
    for (benchmark, model), counts in results.items():
        key = f"{benchmark}/{model}"
        total = counts["total_episodes"]
        serializable[key] = {
            "benchmark": benchmark,
            "model": model,
            "total_episodes": total,
            "channels": {
                "decision": {"count": counts["decision"], "coverage": counts["decision"] / total if total > 0 else 0},
                "trajectory": {"count": counts["trajectory"], "coverage": counts["trajectory"] / total if total > 0 else 0},
                "evidence_contacts": {"count": counts["evidence_contacts"], "coverage": counts["evidence_contacts"] / total if total > 0 else 0},
                "reasoning_text": {"count": counts["reasoning_text"], "coverage": counts["reasoning_text"] / total if total > 0 else 0},
            },
        }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nExported to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit divergence channel coverage across DFAH-Bench replay logs"
    )
    parser.add_argument(
        "--run-logs-dir",
        type=Path,
        default=None,
        help="Path to run_logs directory (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Export results as JSON to this path",
    )
    args = parser.parse_args()

    run_logs_dir = args.run_logs_dir or find_run_logs_dir()
    print(f"Scanning: {run_logs_dir}")

    results = scan_run_logs(run_logs_dir)
    print_matrix(results)

    if args.output:
        export_json(results, args.output)


if __name__ == "__main__":
    main()
