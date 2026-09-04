from __future__ import annotations

import pytest

from dfah import GatePolicy, Replay, ReplayMode, Report
from dfah.demo import make_toy_agent
from dfah.exceptions import ConfigurationError, GateViolationError


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
