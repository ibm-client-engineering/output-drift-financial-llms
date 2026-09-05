"""A run directory without a persisted report fails closed instead of parsing other JSON."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dfah import GatePolicy, Replay, Report
from dfah.cli import app
from dfah.demo import make_toy_agent


def test_run_directory_without_report_fails_closed(tmp_path):
    candidate, suite, _tools, _calls, _tool_calls = make_toy_agent()
    run_dir = tmp_path / "run"
    Replay(suite=suite, replays=2, out=run_dir, gate=GatePolicy()).run(candidate)
    for path in (run_dir / "reports").glob("*.json"):
        path.unlink()
    # The run plan and the gate record are still JSON files in the run root.
    assert (run_dir / "run-plan.json").exists()
    assert list((run_dir / "gates").glob("*.json"))

    with pytest.raises(FileNotFoundError, match="reports"):
        Report.from_json(run_dir)

    analyze = CliRunner().invoke(app, ["analyze", str(run_dir)])
    assert analyze.exit_code == 2, analyze.output
    assert "reports" in analyze.output

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(GatePolicy().model_dump_json(), encoding="utf-8")
    gate = CliRunner().invoke(app, ["gate", str(run_dir), str(policy_path)])
    assert gate.exit_code == 2, gate.output
    assert "reports" in gate.output
