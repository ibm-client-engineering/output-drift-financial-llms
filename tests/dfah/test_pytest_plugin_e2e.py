from __future__ import annotations

from pathlib import Path

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
    failed.assert_outcomes(passed=1, failed=1)
    failed.stdout.fnmatch_lines(["*DFAH gate failed: observed_groups*"])


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
