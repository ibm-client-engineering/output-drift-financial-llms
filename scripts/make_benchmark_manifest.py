#!/usr/bin/env python3
"""Create a frozen release manifest with hashes for benchmark artifacts.

Walks run logs and data/dfah_bench/, SHA-256 hashes every file, and writes
a manifest for reproducible verification.

Usage:
    python scripts/make_benchmark_manifest.py
    python scripts/make_benchmark_manifest.py --output data/dfah_bench/manifest.json
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_and_hash(directory: Path) -> dict:
    """Walk a directory and hash all files."""
    files = {}
    if not directory.exists():
        return files
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            rel = str(path.relative_to(directory))
            files[rel] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Create frozen release manifest for DFAH-Bench artifacts"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/dfah_bench/manifest.json"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    # Directories to include
    sources = {
        "run_logs": repo_root / "econometrics" / "benchmarks" / "results" / "run_logs",
        "provenance_bundles": repo_root / "data" / "provenance_bundles",
        "bench_source": repo_root / "bench",
    }

    manifest = {
        "version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": {},
    }

    total_files = 0
    total_bytes = 0

    for source_name, source_dir in sources.items():
        print(f"Scanning {source_name}: {source_dir}")
        files = walk_and_hash(source_dir)
        manifest["sources"][source_name] = {
            "path": str(source_dir.relative_to(repo_root)),
            "file_count": len(files),
            "total_bytes": sum(f["size_bytes"] for f in files.values()),
            "files": files,
        }
        total_files += len(files)
        total_bytes += manifest["sources"][source_name]["total_bytes"]
        print(f"  {len(files)} files, {manifest['sources'][source_name]['total_bytes']:,} bytes")

    manifest["summary"] = {
        "total_files": total_files,
        "total_bytes": total_bytes,
    }

    # Write manifest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {args.output}")
    print(f"Total: {total_files} files, {total_bytes:,} bytes")


if __name__ == "__main__":
    main()
