#!/usr/bin/env python3
"""
Structured Run Logging for Benchmark Experiments

Writes per-run JSON log entries and batch metadata files.
Used by run_unified_benchmark.py, perturbation_runner.py, temperature_sweep.py.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _hash_tool_output(output: Any) -> str:
    """SHA-256 hash of a tool output for lightweight logging."""
    serialized = json.dumps(output, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def _results_root() -> Path:
    """Root directory for all results."""
    return Path(__file__).parent / "results"


class RunLogger:
    """Writes structured per-run JSON logs.

    Lightweight logs go to:
        results/run_logs/{benchmark}/{model}/case_{id}_run_{id}.json

    Full logs (for failures) go to:
        results/run_logs/{benchmark}/{model}/case_{id}_run_{id}_full.json
    """

    def __init__(self, benchmark: str, model: str):
        self.benchmark = benchmark
        self.model_slug = model.replace(":", "_").replace("/", "_")
        self.log_dir = _results_root() / "run_logs" / benchmark / self.model_slug
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logged_count = 0
        self.total_bytes = 0
        self.failure_count = 0

    def log_run(
        self,
        case_id: str,
        run_id: int,
        seed: int,
        temperature: float,
        tool_sequence: List[str],
        tool_outputs: List[Any],
        decision_output: str,
        deterministic: bool,
        faithfulness_score: float,
        runtime_seconds: float,
        extra: Optional[Dict] = None,
    ) -> Path:
        """Write a structured run log entry.

        Returns path to the written log file.
        """
        tool_output_hashes = [_hash_tool_output(o) for o in tool_outputs]

        entry = {
            "model": self.model_slug,
            "benchmark": self.benchmark,
            "case_id": str(case_id),
            "run_id": run_id,
            "seed": seed,
            "temperature": temperature,
            "timestamp": datetime.now().isoformat(),
            "tool_sequence": tool_sequence,
            "tool_output_hashes": tool_output_hashes,
            "decision_output": decision_output,
            "deterministic": deterministic,
            "faithfulness_score": faithfulness_score,
            "runtime_seconds": round(runtime_seconds, 3),
        }
        if extra:
            entry["extra"] = extra

        safe_case_id = str(case_id).replace("/", "_").replace(" ", "_")
        basename = f"case_{safe_case_id}_run_{run_id}"
        log_path = self.log_dir / f"{basename}.json"

        data = json.dumps(entry, indent=2, default=str)
        log_path.write_text(data)
        self.logged_count += 1
        self.total_bytes += len(data)

        # Write full log for non-deterministic or low-faithfulness runs.
        # Guard: never overwrite an existing full log that has tool outputs
        # with one that has none — re-log passes (e.g. determinism
        # corrections) must not destroy the evidence channel.
        if not deterministic or faithfulness_score < 0.8:
            full_path = self.log_dir / f"{basename}_full.json"
            if not tool_outputs and full_path.exists():
                try:
                    existing = json.loads(full_path.read_text())
                    if existing.get("tool_outputs"):
                        return log_path  # keep the richer existing full log
                except (json.JSONDecodeError, OSError):
                    pass  # unreadable existing file — overwrite below
            full_entry = dict(entry)
            full_entry["tool_outputs"] = tool_outputs
            full_data = json.dumps(full_entry, indent=2, default=str)
            full_path.write_text(full_data)
            self.total_bytes += len(full_data)
            self.failure_count += 1

        return log_path

    def summary(self) -> str:
        """Return a summary string."""
        return (
            f"{self.logged_count} logs written, "
            f"{self.total_bytes:,} bytes total, "
            f"{self.failure_count} failures (full logs)"
        )


class BatchMetadata:
    """Writes batch metadata JSON at experiment start/end.

    Stored at: results/metadata/{timestamp}_batch.json
    """

    def __init__(
        self,
        models: List[str],
        benchmarks: List[str],
        cases_per_benchmark: int,
        runs_per_case: int,
        temperatures: Optional[List[float]] = None,
    ):
        self.meta_dir = _results_root() / "metadata"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now().isoformat()
        self.data = {
            "models": models,
            "benchmarks": benchmarks,
            "cases_per_benchmark": cases_per_benchmark,
            "runs_per_case": runs_per_case,
            "temperatures": temperatures or [0.0],
            "git_commit": _get_git_commit(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "start_time": self.start_time,
            "end_time": None,
        }

    def finalize(self) -> Path:
        """Write metadata file with end_time. Returns the file path."""
        self.data["end_time"] = datetime.now().isoformat()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.meta_dir / f"{ts}_batch.json"
        path.write_text(json.dumps(self.data, indent=2, default=str))
        return path
