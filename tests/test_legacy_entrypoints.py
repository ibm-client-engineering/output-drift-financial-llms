"""Compatibility checks for documented workshop entry points."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import run_dfah_demo
import run_evaluation
from scripts.workshop import make_tables, plot_results
from scripts.workshop import run_dfah_demo as canonical_demo

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    [
        "run_evaluation.py",
        "run_dfah_demo.py",
        "make_tables.py",
        "plot_results.py",
        "DFAH.md",
        "COMMUNITY_FINDINGS.md",
    ],
)
def test_documented_root_paths_remain_available(relative_path: str) -> None:
    assert (REPO_ROOT / relative_path).is_file()


@pytest.mark.parametrize("script", ["run_evaluation.py", "run_dfah_demo.py"])
def test_runner_help_works_outside_repository(tmp_path: Path, script: str) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / script), "--help"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_runner_output_paths_remain_at_repository_root() -> None:
    assert run_evaluation.BASE == REPO_ROOT
    assert run_evaluation.DATA_DIR == REPO_ROOT / "data"
    assert run_evaluation.RESULTS_DIR == REPO_ROOT / "results"
    assert run_evaluation.TRACES_DIR == REPO_ROOT / "traces"
    assert run_dfah_demo.OUTPUT_DIR == REPO_ROOT / "dfah_results"
    assert canonical_demo.OUTPUT_DIR == REPO_ROOT / "dfah_results"
    assert run_dfah_demo.main is canonical_demo.main


def test_canonical_demo_help_works() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.workshop.run_dfah_demo", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--n-cases" in completed.stdout


def test_table_help_works_without_results(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "make_tables.py"), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--rows-only" in completed.stdout

    canonical = subprocess.run(
        [sys.executable, "-m", "scripts.workshop.make_tables", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert canonical.returncode == 0, canonical.stderr
    assert "--rows-only" in canonical.stdout


def test_rows_only_export_supports_booktabs() -> None:
    content = """\\begin{tabular}{lr}
\\toprule
Name & Value \\\\
\\midrule
alpha & 1 \\\\
beta & 2 \\\\
\\bottomrule
\\end{tabular}
"""

    assert make_tables._extract_latex_rows(content) == "alpha & 1 \\\\\nbeta & 2 \\\\\n"


def test_plot_help_works_without_results(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "plot_results.py"), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "workshop figures" in completed.stdout
    assert plot_results.TASKS == ["rag", "summary", "sql"]

    canonical = subprocess.run(
        [sys.executable, "-m", "scripts.workshop.plot_results", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert canonical.returncode == 0, canonical.stderr
    assert "workshop figures" in canonical.stdout
