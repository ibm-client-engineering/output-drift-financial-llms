#!/usr/bin/env python3
"""Build provenance chains from existing replay run logs.

Scans econometrics/benchmarks/results/run_logs/ and creates a hash chain
per (benchmark, model) group. Each run log becomes one chain event.
Issues an Ed25519 certificate per chain and exports bundles.

Usage:
    python scripts/build_provenance_chains.py
    python scripts/build_provenance_chains.py --benchmark compliance --model qwen2.5_7b-instruct
    python scripts/build_provenance_chains.py --output-dir data/provenance_bundles
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.provenance.chain import Chain
from bench.provenance.certificate import generate_keypair, issue_certificate
from bench.provenance.verify import export_bundle, verify_bundle


def find_run_logs_dir() -> Path:
    """Locate the run_logs directory."""
    repo_root = Path(__file__).resolve().parent.parent
    d = repo_root / "econometrics" / "benchmarks" / "results" / "run_logs"
    if d.is_dir():
        return d
    print(f"ERROR: Run logs directory not found at {d}", file=sys.stderr)
    sys.exit(1)


def scan_run_logs(
    run_logs_dir: Path,
    benchmark_filter: str = None,
    model_filter: str = None,
) -> Dict[Tuple[str, str], List[Path]]:
    """Group run log files by (benchmark, model)."""
    groups: Dict[Tuple[str, str], List[Path]] = defaultdict(list)

    for benchmark_dir in sorted(run_logs_dir.iterdir()):
        if not benchmark_dir.is_dir():
            continue
        benchmark = benchmark_dir.name
        if benchmark_filter and benchmark != benchmark_filter:
            continue

        for model_dir in sorted(benchmark_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            if model_filter and model != model_filter:
                continue

            for log_file in sorted(model_dir.glob("case_*_run_*.json")):
                if "_full" in log_file.name:
                    continue
                groups[(benchmark, model)].append(log_file)

    return dict(groups)


def build_chain(
    benchmark: str,
    model: str,
    log_files: List[Path],
) -> Tuple[Chain, int]:
    """Build a hash chain from a list of run log files.

    Returns (chain, events_added).
    """
    chain_id = f"dfah-bench/{benchmark}/{model}"
    chain = Chain(chain_id, model)

    events_added = 0
    for log_file in log_files:
        try:
            with open(log_file) as f:
                data = json.load(f)

            timestamp = data.get("timestamp", chain.created_at)
            # Ensure non-decreasing timestamps
            if chain.events and timestamp < chain.events[-1].timestamp:
                timestamp = chain.events[-1].timestamp

            event_type = f"{benchmark}.{model}.replay"
            chain.append(event_type, data, timestamp=timestamp)
            events_added += 1
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: Skipping {log_file.name}: {e}", file=sys.stderr)

    return chain, events_added


def main():
    parser = argparse.ArgumentParser(
        description="Build provenance chains from existing replay run logs"
    )
    parser.add_argument("--benchmark", default=None, help="Filter by benchmark")
    parser.add_argument("--model", default=None, help="Filter by model")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/provenance_bundles"),
        help="Output directory for bundles",
    )
    parser.add_argument("--run-logs-dir", type=Path, default=None)
    args = parser.parse_args()

    run_logs_dir = args.run_logs_dir or find_run_logs_dir()
    output_dir = args.output_dir

    print(f"Scanning: {run_logs_dir}")
    groups = scan_run_logs(run_logs_dir, args.benchmark, args.model)

    if not groups:
        print("No run logs found.")
        return

    print(f"Found {len(groups)} benchmark-model groups\n")

    # Generate one keypair for all chains in this batch
    private_key, public_key = generate_keypair()

    total_chains = 0
    total_events = 0
    all_verified = True

    for (benchmark, model), log_files in sorted(groups.items()):
        print(f"  {benchmark}/{model}: {len(log_files)} run logs...", end=" ")

        chain, events = build_chain(benchmark, model, log_files)
        if events == 0:
            print("SKIP (no valid events)")
            continue

        # Issue certificate
        cert_result = issue_certificate(
            chain, private_key,
            metadata={"benchmark": benchmark, "model": model},
        )

        # Export bundle
        bundle = export_bundle(chain, cert_result)

        # Verify
        result = verify_bundle(bundle)
        if not result.valid:
            print(f"FAIL ({result.errors})")
            all_verified = False
            continue

        # Write bundle
        bundle_dir = output_dir / benchmark / model
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        with open(bundle_path, "w") as f:
            json.dump(bundle, f, indent=2)

        print(f"{events} events, verified ✓")
        total_chains += 1
        total_events += events

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Chains built:  {total_chains}")
    print(f"Total events:  {total_events}")
    print(f"All verified:  {'YES' if all_verified else 'NO'}")
    print(f"Output:        {output_dir}")


if __name__ == "__main__":
    main()
