"""Small, deterministic agent used by the installed-package quickstart."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from ._frozen import FrozenJson
from .agents import AgentResult, CallableAgent, RunContext, agent
from .exceptions import ConfigurationError
from .models import Case, Usage, WireRequest
from .replay import build_manifest
from .suite import Suite, ToolSpec
from .tools import ToolRegistry


def make_toy_agent() -> tuple[
    CallableAgent,
    Suite,
    ToolRegistry,
    dict[str, int],
    dict[str, int],
]:
    """Build a no-network agent, its versioned suite, and local call counters."""

    risk_tool = ToolSpec(
        name="read_risk_tier",
        input_schema={
            "type": "object",
            "properties": {"risk_tier": {"type": "string"}},
            "required": ["risk_tier"],
            "additionalProperties": False,
        },
    )
    suite = Suite(
        suite_id="toy-risk",
        suite_version="1.0.0",
        decisions=("escalate", "dismiss"),
        cases=(
            Case(
                case_id="RISK-001",
                artifact_case_id="CASE-001",
                input={"risk_tier": "high"},
            ),
            Case(
                case_id="RISK-002",
                artifact_case_id="CASE-002",
                input={"risk_tier": "low"},
            ),
        ),
        tools=(risk_tool,),
        description="Deterministic no-network tool-session example.",
    )
    tools = ToolRegistry()
    tool_calls = {"count": 0}

    @tools.tool(risk_tool)
    def read_risk_tier(*, risk_tier: str) -> str:
        tool_calls["count"] += 1
        return risk_tier

    parameters: dict[str, FrozenJson] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = build_manifest(
        suite,
        provider="toy",
        model="deterministic-policy-v1",
        adapter="dfah.demo",
        adapter_version="1.0.0",
        implementation_hash=implementation_hash,
        request_parameters=parameters,
    )
    agent_calls = {"count": 0}

    @agent(manifest=manifest, tools=tools, suite=suite)
    async def toy_agent(case: Case, context: RunContext) -> AgentResult:
        assert context.tools is not None
        agent_calls["count"] += 1
        case_input = case.input
        if not isinstance(case_input, Mapping):
            raise ConfigurationError("toy case input must be an object")
        risk_tier = case_input.get("risk_tier")
        if not isinstance(risk_tier, str):
            raise ConfigurationError("toy case risk_tier must be a string")
        risk = await context.tools.call(
            "read_risk_tier",
            risk_tier=risk_tier,
        )
        decision = "ESCALATE" if risk == "high" else "DISMISS"
        payload: dict[str, FrozenJson] = {
            "model": "deterministic-policy-v1",
            **parameters,
            "case_id": case.case_id,
        }
        return AgentResult(
            output_text=f"DECISION: {decision}",
            trajectory=context.tools.trajectory(),
            wire_request=WireRequest.from_payload(
                provider="toy",
                model="deterministic-policy-v1",
                payload=payload,
                parameters=parameters,
                adapter="dfah.demo",
                adapter_version="1.0.0",
            ),
            usage=Usage(input_tokens=0, output_tokens=0),
            cost_usd=0.0,
        )

    return toy_agent, suite, tools, agent_calls, tool_calls


toy_agent, toy_suite, _toy_tools, toy_agent_calls, toy_tool_calls = make_toy_agent()

__all__ = [
    "make_toy_agent",
    "toy_agent",
    "toy_agent_calls",
    "toy_suite",
    "toy_tool_calls",
]
