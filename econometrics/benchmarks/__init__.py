"""
V3 Financial Agent Benchmarks

Three benchmark tasks for evaluating determinism and faithfulness
of tool-using LLM agents in financial applications.

Tasks:
- compliance_triage: Regulatory alert escalation decisions
- portfolio_constraint: Investment constraint validation
- dataops_exception: Data quality exception handling

Usage:
    python -m econometrics.benchmarks.run_all

Reference:
    ICLR 2026 FinAI Workshop - "Replayable Financial Agents"
"""

__version__ = "1.0.0"
__author__ = "IBM Client Engineering"

from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)
