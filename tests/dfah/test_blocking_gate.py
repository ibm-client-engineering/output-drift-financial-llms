from __future__ import annotations

import pytest

from dfah import GatePolicy, Replay, ReplayMode, Report
from dfah._canonical import sha256
from dfah.demo import make_toy_agent
from dfah.exceptions import ConfigurationError, GateViolationError
from dfah.gate import GateRecord


@pytest.mark.parametrize("mode", [ReplayMode.BLOCKING, "blocking"])
def test_blocking_without_a_policy_is_rejected_before_dispatch(tmp_path, mode):
    candidate, suite, _tools, calls, _tool_calls = make_toy_agent()
    run_dir = tmp_path / "invalid-run"
    with pytest.raises(ConfigurationError, match="requires an explicit GatePolicy"):
        Replay(suite=suite, mode=mode, out=run_dir).run(candidate)
    assert calls["count"] == 0
    assert not run_dir.exists()


def test_blocking_policy_failure_preserves_verified_evidence(tmp_path):
    candidate, suite, _tools, calls, _tool_calls = make_toy_agent()
    runner = Replay(
        suite=suite,
        mode=ReplayMode.BLOCKING,
        gate=GatePolicy(min_observed_groups=3),
        replays=2,
        out=tmp_path / "run",
    )
    with pytest.raises(GateViolationError, match="observed_groups"):
        runner.run(candidate)
    assert calls["count"] == 4
    report = Report.from_json(tmp_path / "run")
    assert report.artifacts_verified
    assert report.observed_groups == 2
    # The failed evaluation is recorded before the exception is raised.
    record = GateRecord.from_json(tmp_path / "run" / "gates" / f"{report.report_id}.json")
    assert record.mode is ReplayMode.BLOCKING
    assert record.result.passed is False
    assert runner.last_gate_result == record.result


def test_shadow_policy_outcome_is_recorded_and_exposed(tmp_path):
    candidate, suite, _tools, calls, _tool_calls = make_toy_agent()
    policy = GatePolicy(min_observed_groups=3)
    runner = Replay(
        suite=suite,
        mode=ReplayMode.SHADOW,
        gate=policy,
        replays=2,
        out=tmp_path / "run",
    )
    report = runner.run(candidate)
    assert calls["count"] == 4
    assert report.artifacts_verified
    assert runner.last_gate_result is not None
    assert runner.last_gate_result.passed is False
    record = GateRecord.from_json(tmp_path / "run" / "gates" / f"{report.report_id}.json")
    assert record.report_id == report.report_id
    assert record.manifest_hash == report.manifest.hash
    assert record.mode is ReplayMode.SHADOW
    assert record.policy == policy
    assert record.policy_sha256 == sha256(policy)
    assert [check.name for check in record.result.checks if not check.passed] == [
        "observed_groups"
    ]
    # The recorded report still verifies; the gate record lives outside the store.
    assert Report.from_json(tmp_path / "run").report_id == report.report_id


def test_run_without_a_policy_records_no_gate(tmp_path):
    candidate, suite, _tools, _calls, _tool_calls = make_toy_agent()
    runner = Replay(suite=suite, replays=2, out=tmp_path / "run")
    runner.run(candidate)
    assert runner.last_gate_result is None
    assert not (tmp_path / "run" / "gates").exists()
