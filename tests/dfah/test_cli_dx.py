from __future__ import annotations

import importlib
import json
import sys

import pytest
from typer.testing import CliRunner

from dfah import (
    AgentResult,
    ChannelState,
    Suite,
    Trajectory,
    WireRequest,
    agent,
    build_manifest,
)
from dfah.cli import app


def test_cli_validates_a_builtin_suite_by_name():
    result = CliRunner().invoke(app, ["suites", "validate", "compliance-v1"])
    assert result.exit_code == 0, result.output
    assert "valid compliance-v1@1.0.0" in result.output


def test_cli_infers_an_agent_bound_suite_and_safe_default_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--agent",
            "dfah.demo:toy_agent",
            "--replays",
            "2",
            "--episode-timeout-s",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "run=.dfah/runs/toy-risk-" in result.output
    assert "status=complete" in result.output
    assert "gap=0.000" in result.output
    run_dirs = list((tmp_path / ".dfah" / "runs").glob("toy-risk-*"))
    assert len(run_dirs) == 1
    plan = json.loads((run_dirs[0] / "run-plan.json").read_text(encoding="utf-8"))
    assert plan["episode_timeout_s"] == 5.0


def test_cli_renders_unavailable_metrics_and_eligibility_reasons(tmp_path, monkeypatch):
    suite = Suite.load("compliance-v1")
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="wire-unstable-v1",
        adapter="tests.cli-wire-unstable",
        request_parameters=parameters,
        git_sha="f" * 40,
    )

    @agent(manifest=manifest, suite=suite)
    async def candidate(case, context):
        payload = {
            "case_id": case.case_id,
            "replay_nonce": context.replay_index,
            **parameters,
        }
        return AgentResult(
            output_text="DECISION: ESCALATE",
            trajectory=Trajectory(state=ChannelState.OBSERVED_EMPTY),
            wire_request=WireRequest.from_payload(
                provider="fake",
                model="wire-unstable-v1",
                payload=payload,
                parameters=parameters,
                adapter="tests.cli-wire-unstable",
            ),
        )

    cli_module = importlib.import_module("dfah.cli.main")
    monkeypatch.setattr(cli_module, "_load_object", lambda _: candidate)
    run_dir = tmp_path / "ineligible"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--agent",
            "ignored:agent",
            "--replays",
            "2",
            "--out",
            str(run_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DAR=— TARseq=— gap=— flags/100=— status=partial" in result.output
    assert "eligible_groups=0/2" in result.output
    assert "eligible_episodes=0/4" in result.output
    assert "reasons=wire_payload_mismatch=2" in result.output

    analyzed = CliRunner().invoke(app, ["analyze", str(run_dir)])
    assert analyzed.exit_code == 0, analyzed.output
    assert "DAR=— TARseq=— gap=— flags/100=—" in analyzed.output
    assert "artifacts=verified" in analyzed.output
    assert "reasons=wire_payload_mismatch=2" in analyzed.output


def test_check_agent_cli_accepts_a_bounded_episode_timeout():
    result = CliRunner().invoke(
        app,
        [
            "check-agent",
            "--agent",
            "dfah.demo:toy_agent",
            "--episode-timeout-s",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "planned calls: at most 4" in result.output


def test_expected_cli_error_is_concise_and_has_no_traceback(monkeypatch, capsys):
    cli_module = importlib.import_module("dfah.cli.main")
    monkeypatch.setattr(
        sys,
        "argv",
        ["dfah", "suites", "validate", "does-not-exist"],
    )
    with pytest.raises(SystemExit) as error:
        cli_module.main()
    assert error.value.code == 1
    captured = capsys.readouterr()
    rendered = captured.out + captured.err
    assert "Error: unknown suite" in rendered
    assert "Traceback" not in rendered
    assert "request_parameters" not in rendered
