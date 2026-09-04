from __future__ import annotations

from pathlib import Path

import pytest

from dfah import (
    AgentResult,
    ChannelState,
    GatePolicy,
    Replay,
    Suite,
    Trajectory,
    WireRequest,
    agent,
    build_manifest,
)

pytest_plugins = ("pytester",)


def _verified_run(tmp_path: Path):
    suite = Suite.load("compliance-v1")
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="plugin-v1",
        adapter="tests.plugin",
        request_parameters=parameters,
        git_sha="f" * 40,
    )

    @agent(manifest=manifest, suite=suite)
    async def candidate(case, context):
        return AgentResult(
            output_text="DECISION: ESCALATE",
            trajectory=Trajectory(state=ChannelState.OBSERVED_EMPTY),
            wire_request=WireRequest.from_payload(
                provider="fake",
                model="plugin-v1",
                payload={"model": "plugin-v1", **parameters},
                parameters=parameters,
                adapter="tests.plugin",
            ),
        )

    run_dir = tmp_path / "verified-run"
    report = Replay(suite=suite, replays=2, out=run_dir).run(candidate)
    return run_dir, report


def _plugin_args() -> tuple[str, ...]:
    return ("-p", "no:dfah", "-p", "dfah.pytest_plugin")


def test_pytest_plugin_skips_without_a_report(pytester):
    pytester.makepyfile(
        """
        def test_release_evidence(dfah_report):
            assert dfah_report.artifacts_verified
        """
    )
    result = pytester.runpytest(*_plugin_args(), "-q")
    result.assert_outcomes(skipped=1)


def test_pytest_plugin_loads_verified_run_and_applies_policy(pytester, tmp_path):
    run_dir, _report = _verified_run(tmp_path)
    passing = tmp_path / "passing-policy.json"
    passing.write_text(
        GatePolicy(
            min_dar=1.0,
            min_tar_seq=1.0,
            min_observed_groups=2,
            require_complete=True,
        ).model_dump_json(),
        encoding="utf-8",
    )
    pytester.makepyfile(
        """
        def test_release_evidence(dfah_report):
            assert dfah_report.artifacts_verified
            assert dfah_report.dar == 1.0

        def test_release_gate(dfah_gate):
            dfah_gate().raise_for_failures()
        """
    )
    result = pytester.runpytest(
        *_plugin_args(),
        "--dfah-report",
        str(run_dir),
        "--dfah-policy",
        str(passing),
        "-q",
    )
    result.assert_outcomes(passed=2)

    failing = tmp_path / "failing-policy.json"
    failing.write_text(GatePolicy(min_observed_groups=3).model_dump_json(), encoding="utf-8")
    failed = pytester.runpytest(
        *_plugin_args(),
        "--dfah-report",
        str(run_dir),
        "--dfah-policy",
        str(failing),
        "-q",
    )
    assert failed.ret == pytest.ExitCode.TESTS_FAILED
    assert "DFAH gate failed: observed_groups" in failed.stdout.str() + failed.stderr.str()


@pytest.mark.parametrize("uses_fixture", [True, False])
def test_requested_policy_is_enforced_without_an_explicit_gate_test(
    pytester, tmp_path, uses_fixture
):
    run_dir, _report = _verified_run(tmp_path)
    policy = tmp_path / "failing-policy.json"
    policy.write_text(GatePolicy(min_observed_groups=3).model_dump_json(), encoding="utf-8")
    argument = "dfah_report" if uses_fixture else ""
    pytester.makepyfile(f"def test_release({argument}):\n    assert True\n")
    result = pytester.runpytest(
        *_plugin_args(), "--dfah-report", str(run_dir), "--dfah-policy", str(policy), "-q"
    )
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert "DFAH gate failed: observed_groups" in result.stdout.str() + result.stderr.str()


@pytest.mark.parametrize("required_groups", [2, 3])
def test_requested_policy_runs_even_when_no_tests_are_collected(
    pytester, tmp_path, required_groups
):
    run_dir, _report = _verified_run(tmp_path)
    policy = tmp_path / "policy.json"
    policy.write_text(
        GatePolicy(min_observed_groups=required_groups).model_dump_json(), encoding="utf-8"
    )
    pytester.makepyfile("# Intentionally no tests.\n")
    result = pytester.runpytest(
        *_plugin_args(), "--dfah-report", str(run_dir), "--dfah-policy", str(policy), "-q"
    )
    if required_groups == 2:
        assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED
        assert "DFAH policy passed" in result.stdout.str()
    else:
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        assert "DFAH gate failed" in result.stdout.str() + result.stderr.str()


def test_requested_policy_requires_a_report_even_without_tests(pytester, tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(GatePolicy().model_dump_json(), encoding="utf-8")
    result = pytester.runpytest(*_plugin_args(), "--dfah-policy", str(policy), "-q")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "--dfah-policy requires --dfah-report" in result.stderr.str()


@pytest.mark.parametrize(
    ("suffix", "content"),
    [
        (".json", "{broken-json"),
        (".json", '{"min_dar": "private-input-must-not-be-echoed"}'),
        (".yaml", "min_dar: ["),
        (".json", None),
    ],
)
def test_requested_policy_rejects_invalid_input_before_collection(
    pytester, tmp_path, suffix, content
):
    run_dir, _report = _verified_run(tmp_path)
    policy = tmp_path / f"invalid-policy{suffix}"
    if content is not None:
        policy.write_text(content, encoding="utf-8")
    result = pytester.runpytest(
        *_plugin_args(), "--dfah-report", str(run_dir), "--dfah-policy", str(policy), "-q"
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    output = result.stdout.str() + result.stderr.str()
    assert "valid policy and verified report" in output
    assert "private-input-must-not-be-echoed" not in output


def test_requested_policy_rejects_missing_report_artifacts(pytester, tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(GatePolicy().model_dump_json(), encoding="utf-8")
    result = pytester.runpytest(
        *_plugin_args(),
        "--dfah-report",
        str(tmp_path / "missing-run"),
        "--dfah-policy",
        str(policy),
        "-q",
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "valid policy and verified report" in result.stderr.str()


def test_pytest_plugin_rejects_a_detached_report(pytester, tmp_path):
    _run_dir, report = _verified_run(tmp_path)
    standalone = tmp_path / "standalone-report.json"
    report.to_json(standalone)
    pytester.makepyfile(
        """
        def test_release_evidence(dfah_report):
            assert dfah_report.artifacts_verified
        """
    )
    result = pytester.runpytest(*_plugin_args(), "--dfah-report", str(standalone), "-q")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*standalone report has no episode store*"])
