"""Exercise the installed-package example and verify its recorded replay evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from dfah import Gate, GatePolicy, PathVariationKind, Report, ToolExecutionState
from dfah.store import FileStore

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "dfah_gate_loop.py"


def run_example(out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLE), "--out", str(out)],
        cwd=out.parent,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_gate_loop_records_a_failed_candidate_then_a_passing_correction(tmp_path: Path) -> None:
    out = tmp_path / "review"
    completed = run_example(out)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert [row["passed"] for row in summary] == [False, True]
    assert [row["adapter_version"] for row in summary] == ["1.0.0", "1.0.1"]
    assert "01-varying-path: FAIL" in completed.stdout
    assert "02-fixed-path: PASS" in completed.stdout
    policy = GatePolicy.load(out / "policy.json")
    reports = []
    for row in summary:
        run_dir = out / row["candidate"]
        report = Report.from_json(run_dir)
        reports.append(report)
        assert report.artifacts_verified
        assert report.episodes_completed == report.episodes_eligible == 4
        assert report.observed_groups == 2
        assert Gate(policy).evaluate(report).passed is row["passed"]
        assert row["manifest_sha256"] == report.manifest.hash
        assert row["episode_root_sha256"] == report.episode_artifact_root_sha256
        assert "<!doctype html>" in (out / row["report_html"]).read_text().lower()
        saved_gate = json.loads((run_dir / "gate.json").read_text())
        assert saved_gate["passed"] is row["passed"]
        episodes = FileStore(run_dir, create=False).list(manifest_hash=report.manifest.hash)
        assert len(episodes) == 4
        decisions = {}
        for episode in episodes:
            assert len(episode.trajectory.tool_calls) == 2
            for call in episode.trajectory.tool_calls:
                assert call.execution_state is ToolExecutionState.EXECUTED
                assert call.arguments_hash is not None and call.output_hash is not None
                assert call.latency_ms is not None
            assert episode.decision is not None
            decisions[episode.case_id] = episode.decision.label
        assert decisions == {"CASE-001": "proceed", "CASE-002": "review"}

    first, corrected = reports
    assert first.manifest.hash != corrected.manifest.hash
    assert first.manifest.implementation_hash != corrected.manifest.implementation_hash
    assert first.episode_artifact_root_sha256 != corrected.episode_artifact_root_sha256
    assert first.run_plan_sha256 != corrected.run_plan_sha256
    assert first.manifest.fixture_hash == corrected.manifest.fixture_hash
    assert first.manifest.tool_schema_hash == corrected.manifest.tool_schema_hash
    assert first.dar == corrected.dar == 1.0
    assert first.tar is not None and corrected.tar is not None
    assert first.tar.seq == 0.5 and corrected.tar.seq == 1.0
    assert first.flagged_groups == 2 and corrected.flagged_groups == 0
    assert first.flags_per_100_cases == 100.0 and corrected.flags_per_100_cases == 0.0
    assert set(summary[0]["failed_checks"]) == {"tar_seq", "gap", "flags_per_100_cases"}
    assert summary[1]["failed_checks"] == []
    assert all(
        row.path_variation_kind is PathVariationKind.ORDER_ONLY for row in first.case_reports
    )


def test_gate_loop_refuses_to_overwrite_a_previous_review(tmp_path: Path) -> None:
    out = tmp_path / "review"
    completed = run_example(out)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    before = {
        path.relative_to(out): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in out.rglob("*")
        if path.is_file()
    }
    repeated = run_example(out)
    assert repeated.returncode == 2
    assert "output directory already exists" in repeated.stderr
    after = {
        path.relative_to(out): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in out.rglob("*")
        if path.is_file()
    }
    assert after == before
