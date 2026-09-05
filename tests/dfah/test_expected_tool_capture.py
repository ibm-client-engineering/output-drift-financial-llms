"""check_agent capture semantics: empty paths stay valid unless expectations say otherwise."""

from __future__ import annotations

import hashlib
import importlib

import pytest
from typer.testing import CliRunner

from dfah import (
    AgentResult,
    Case,
    ChannelState,
    ConformanceStatus,
    RunContext,
    Suite,
    ToolRegistry,
    ToolSpec,
    Trajectory,
    WireRequest,
    agent,
    build_manifest,
)
from dfah.cli import app
from dfah.demo import make_toy_agent
from dfah.exceptions import ConfigurationError
from dfah.testing import check_agent


def _check(result, name):
    return next(check for check in result.checks if check.name == name)


def _two_tool_agent(*, record_status: bool, calls: dict[str, int] | None = None):
    """Two declared tools; ``read_rule`` is always invoked directly, bypassing capture."""

    status_spec = ToolSpec(
        name="read_status",
        input_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        },
    )
    rule_spec = ToolSpec(
        name="read_rule", input_schema={"type": "object", "additionalProperties": False}
    )
    suite = Suite(
        suite_id="capture-conformance",
        suite_version="1.0.0",
        decisions=("proceed", "review"),
        cases=(
            Case(case_id="C1", input={"status": "ready"}),
            Case(case_id="C2", input={"status": "waiting"}),
        ),
        tools=(status_spec, rule_spec),
    )
    tools = ToolRegistry()

    @tools.tool(status_spec)
    def read_status(*, status: str) -> str:
        return status

    @tools.tool(rule_spec)
    def read_rule() -> str:
        return "ready"

    parameters = {"seed": 42}
    manifest = build_manifest(
        suite,
        provider="local",
        model="status-policy",
        adapter="tests.capture",
        adapter_version="1.0.0" if record_status else "1.0.1",
        implementation_hash=hashlib.sha256(str(record_status).encode()).hexdigest(),
        request_parameters=parameters,
    )
    counter = calls if calls is not None else {"count": 0}

    @agent(manifest=manifest, suite=suite, tools=tools)
    async def candidate(case: Case, context: RunContext) -> AgentResult:
        assert context.tools is not None
        counter["count"] += 1
        status = case.input["status"]  # type: ignore[index]
        if record_status:
            observed = await context.tools.call("read_status", status=status)
        else:
            observed = read_status(status=status)  # bypasses the session
        rule = read_rule()  # always bypasses the session
        decision = "PROCEED" if observed == rule else "REVIEW"
        return AgentResult(
            output_text=f"DECISION: {decision}",
            trajectory=context.tools.trajectory(),
            wire_request=WireRequest.from_payload(
                provider="local",
                model="status-policy",
                adapter="tests.capture",
                adapter_version=manifest.adapter_version,
                payload={"model": "status-policy", **parameters, "case_id": case.case_id},
                parameters=parameters,
            ),
        )

    return candidate


def _zero_tool_agent(suite: Suite):
    parameters = {"seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="fake-v1",
        adapter="tests.zero_tool",
        request_parameters=parameters,
        git_sha="f" * 40,
    )

    @agent(manifest=manifest, suite=suite)
    async def candidate(case: Case, context: RunContext) -> AgentResult:
        return AgentResult(
            output_text=f"DECISION: {suite.decisions[0]}",
            trajectory=Trajectory(state=ChannelState.OBSERVED_EMPTY),
            wire_request=WireRequest.from_payload(
                provider="fake",
                model="fake-v1",
                payload={"case_id": case.case_id, **parameters},
                parameters=parameters,
                adapter="tests.zero_tool",
            ),
        )

    return candidate


def test_observed_empty_paths_remain_valid_without_expectations():
    result = check_agent(_two_tool_agent(record_status=False))
    check = _check(result, "expected_tool_capture")
    assert check.status is ConformanceStatus.SKIP
    assert "pass expected_tools" in check.detail
    assert result.passed
    assert result.selected_case_ids == ("C1", "C2")


def test_expectations_detect_a_full_bypass():
    result = check_agent(
        _two_tool_agent(record_status=False),
        expected_tools={"C1": ["read_status", "read_rule"], "C2": ["read_status"]},
    )
    check = _check(result, "expected_tool_capture")
    assert check.status is ConformanceStatus.FAIL
    assert "C1/replay-0:read_rule" in check.detail
    assert "C2/replay-1:read_status" in check.detail
    assert "invisible to DFAH" in check.detail
    assert not result.passed


def test_expectations_detect_a_partial_bypass_only_for_expected_tools():
    partial = _two_tool_agent(record_status=True)
    failed = check_agent(partial, expected_tools={"C1": ["read_status", "read_rule"]})
    detail = _check(failed, "expected_tool_capture").detail
    assert _check(failed, "expected_tool_capture").status is ConformanceStatus.FAIL
    assert ":read_rule" in detail
    assert ":read_status" not in detail
    # An expectation that omits the bypassed tool cannot see the bypass.
    passed = check_agent(partial, expected_tools={"C1": ["read_status"]})
    assert _check(passed, "expected_tool_capture").status is ConformanceStatus.PASS
    assert passed.passed


def test_expectations_pass_for_captured_calls():
    candidate, _suite, _tools, _calls, _tool_calls = make_toy_agent()
    result = check_agent(
        candidate,
        expected_tools={"CASE-001": ["read_risk_tier"], "CASE-002": ["read_risk_tier"]},
    )
    check = _check(result, "expected_tool_capture")
    assert check.status is ConformanceStatus.PASS
    assert "2 case(s)" in check.detail
    assert result.passed


def test_zero_tool_agent_without_expectations_passes():
    result = check_agent(_zero_tool_agent(Suite.load("conformance-v1")))
    check = _check(result, "expected_tool_capture")
    assert check.status is ConformanceStatus.SKIP
    assert "suite declares" not in check.detail
    assert result.passed


@pytest.mark.parametrize(
    ("expected_tools", "message"),
    [
        ({"C9": ["read_status"]}, "not among the selected"),
        ({"C1": ["explode"]}, "does not declare"),
        ({"C1": []}, "must name a tool"),
    ],
)
def test_invalid_expectations_fail_before_any_agent_call(expected_tools, message):
    calls = {"count": 0}
    with pytest.raises(ConfigurationError, match=message):
        check_agent(
            _two_tool_agent(record_status=True, calls=calls), expected_tools=expected_tools
        )
    assert calls["count"] == 0


def test_cli_expect_tools_option(monkeypatch):
    result = CliRunner().invoke(
        app,
        [
            "check-agent",
            "--agent",
            "dfah.demo:toy_agent",
            "--expect-tools",
            "CASE-001=read_risk_tier",
            "--expect-tools",
            "CASE-002=read_risk_tier",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "selected cases: CASE-001, CASE-002" in result.output
    assert "PASS expected_tool_capture" in result.output

    cli_module = importlib.import_module("dfah.cli.main")
    monkeypatch.setattr(
        cli_module, "_load_object", lambda _: _two_tool_agent(record_status=False)
    )
    bypass = CliRunner().invoke(
        app, ["check-agent", "--agent", "ignored:agent", "--expect-tools", "C1=read_rule"]
    )
    assert bypass.exit_code == 1, bypass.output
    assert "FAIL expected_tool_capture" in bypass.output

    malformed = CliRunner().invoke(
        app, ["check-agent", "--agent", "ignored:agent", "--expect-tools", "nonsense"]
    )
    assert isinstance(malformed.exception, ConfigurationError)
    assert "CASE_ID=tool" in str(malformed.exception)
