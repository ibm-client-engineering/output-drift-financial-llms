"""CLI entry point for bundle verification.

Usage:
    python -m bench.provenance.verify bundle.json
    python -m bench.provenance.verify bundle.json --verbose

Exit codes:
    0 = verification passed
    1 = verification failed
"""

import argparse
import json
import sys

from .verify import verify_bundle


def main():
    parser = argparse.ArgumentParser(
        description="Verify a DFAH-Bench provenance bundle"
    )
    parser.add_argument("bundle", help="Path to bundle JSON file")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed verification results",
    )
    args = parser.parse_args()

    try:
        with open(args.bundle) as f:
            bundle = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    result = verify_bundle(bundle)

    if args.verbose:
        print(f"Chain valid:       {result.chain_valid}")
        print(f"Certificate valid: {result.certificate_valid}")
        print(f"Cross-layer valid: {result.cross_layer_valid}")
        print(f"Overall valid:     {result.valid}")
        if result.errors:
            print("\nErrors:")
            for err in result.errors:
                print(f"  - {err}")
    else:
        if result.valid:
            print("PASS")
        else:
            print("FAIL")
            for err in result.errors:
                print(f"  {err}", file=sys.stderr)

    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
