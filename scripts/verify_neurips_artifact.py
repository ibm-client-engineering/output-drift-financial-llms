#!/usr/bin/env python3
"""Verify the staged NeurIPS reviewer artifact.

This verifier is intentionally lightweight and offline. It checks that the
reviewer artifact contains the files the paper claims, and scans text files for
obvious anonymity leaks such as absolute local paths or internal planning
directories. It does not rerun model experiments.

Usage:
    python3 scripts/verify_neurips_artifact.py
    python3 scripts/verify_neurips_artifact.py --artifact-dir dist/neurips2026_artifact
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "dist" / "neurips2026_artifact"

REQUIRED_PATHS = [
    "CONTENTS.md",
    "LICENSE",
    "requirements.txt",
    "bench/README.md",
    "bench/cards/benchmark_card.md",
    "data/dfah_bench/README.md",
    "data/dfah_bench/manifest.json",
    "paper/neurips2026/main.tex",
    "paper/neurips2026/main.pdf",
    "paper/neurips2026/neurips_2026.sty",
    "paper/neurips2026/README.md",
    "paper/neurips2026/NUMERIC_AUDIT.md",
    "paper/neurips2026/ARTIFACT_READINESS.md",
    "scripts/audit_channel_coverage.py",
    "scripts/build_provenance_chains.py",
    "scripts/compute_accuracy_metric_correlations.py",
    "scripts/compute_bootstrap_cis.py",
    "scripts/compute_dcb_across_case.py",
    "scripts/compute_dfah_accuracy.py",
    "scripts/compute_dfah_metrics.py",
    "scripts/compute_gt_baselines.py",
    "scripts/compute_kappa.py",
    "scripts/compute_task_gap_cis.py",
    "scripts/compute_tool_call_counts.py",
    "scripts/generate_paper_figures.py",
    "scripts/make_benchmark_manifest.py",
    "scripts/n3_subsampling_sensitivity.py",
    "scripts/prepare_neurips_artifact.py",
    "scripts/run_perturbation_experiment.py",
    "scripts/verify_neurips_artifact.py",
    "results/dfah_accuracy_benchmark_level.csv",
    "results/dfah_accuracy_metric_correlations.csv",
    "results/dfah_case_level.csv",
    "results/dfah_dcb_across_case_benchmark.csv",
    "results/dfah_dcb_across_case_model.csv",
    "results/dfah_gt_baselines.csv",
    "results/dfah_kappa_benchmark_level.csv",
    "results/dfah_kappa_model_level.csv",
    "results/dfah_model_accuracy.csv",
    "results/dfah_model_cis.csv",
    "results/dfah_model_level.csv",
    "results/dfah_kill_criterion.csv",
    "results/dfah_skipped_case_groups.csv",
    "results/dfah_task_gap_cis.csv",
    "results/dfah_task_level.csv",
    "results/dfah_tool_call_counts_benchmark.csv",
    "results/dfah_tool_call_counts_model.csv",
    "tests/test_dcb.py",
    "tests/test_ecd.py",
    "tests/test_scdr.py",
    "tests/test_stats.py",
    "tests/test_schema.py",
    "tests/test_canonicalize.py",
    "tests/test_chain.py",
    "tests/test_certificate.py",
    "tests/test_verify.py",
]

FORBIDDEN_SUBSTRINGS = [
    "/" + "Users" + "/",
    "\\" + "Users" + "\\",
    "docs" + "-internal",
    "paper" + "-planning",
    "ai4f" + "-drift-runner-pro",
    "JF" + "DS" + "_R1",
    "JF" + "DS" + "-",
    "Raf" + "fi",
    "Khatcha" + "dourian",
    "khatcha" + "dourian",
    "Rolan" + "do",
    "Fran" + "co",
    "arXiv:" + "2601.15322",
    "arXiv:" + "2511.07585",
]

TEXT_SUFFIXES = {
    ".bib",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify staged NeurIPS artifact")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Artifact directory to verify",
    )
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    failures: list[str] = []

    if not artifact_dir.exists():
        failures.append(f"artifact directory does not exist: {artifact_dir}")
    else:
        for rel_path in REQUIRED_PATHS:
            if not (artifact_dir / rel_path).exists():
                failures.append(f"missing required path: {rel_path}")

        for path in artifact_dir.rglob("*"):
            rel = path.relative_to(artifact_dir).as_posix()
            for forbidden in FORBIDDEN_SUBSTRINGS:
                if forbidden in rel:
                    failures.append(f"forbidden substring in path: {rel}")
            if path.is_file() and is_text_file(path):
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for forbidden in FORBIDDEN_SUBSTRINGS:
                    if forbidden in text:
                        failures.append(f"forbidden substring `{forbidden}` in {rel}")

    if failures:
        print("Artifact verification FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"Artifact verification passed: {artifact_dir}")
    print(f"Checked {len(REQUIRED_PATHS)} required paths and anonymity substrings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
