#!/usr/bin/env python3
"""Replay two prewritten local candidates under one unchanged review policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from dfah import (
    AgentResult,
    CallableAgent,
    Case,
    Gate,
    GatePolicy,
    GateResult,
    Replay,
    Report,
    RunContext,
    Suite,
    ToolRegistry,
    ToolSpec,
    WireRequest,
    agent,
    build_manifest,
)

# These two implementations are supplied in advance. The loop never edits code.
CANDIDATES = (
    ("01-varying-path", "1.0.0", True),
    ("02-fixed-path", "1.0.1", False),
)
POLICY = GatePolicy(
    min_dar=1.0,
    min_tar_seq=1.0,
    max_gap=0.0,
    max_flags_per_100_cases=0.0,
    required_replays=2,
    min_eligible_fraction=1.0,
    min_observed_groups=2,
)


def make_candidate(version: str, *, vary_order: bool) -> CallableAgent:
    """Keep the task and final decision fixed while changing the tool order."""

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
        name="read_rule",
        input_schema={"type": "object", "additionalProperties": False},
    )
    suite = Suite(
        suite_id="local-review-example",
        suite_version="1.0.0",
        decisions=("proceed", "review"),
        cases=(
            Case(case_id="CASE-001", input={"status": "ready"}),
            Case(case_id="CASE-002", input={"status": "waiting"}),
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

    # Pin both the source bytes and the candidate configuration, even outside Git.
    implementation = Path(__file__).read_bytes() + f"{version}:{vary_order}".encode()
    parameters = {"seed": 42}
    manifest = build_manifest(
        suite,
        provider="local",
        model="status-policy",
        adapter="examples.dfah_gate_loop",
        adapter_version=version,
        implementation_hash=hashlib.sha256(implementation).hexdigest(),
        request_parameters=parameters,
    )

    @agent(manifest=manifest, suite=suite, tools=tools)
    async def candidate(case: Case, context: RunContext) -> AgentResult:
        assert context.tools is not None
        assert isinstance(case.input, Mapping)
        status = case.input["status"]
        assert isinstance(status, str)
        order: tuple[str, ...] = ("read_status", "read_rule")
        if vary_order and context.replay_index % 2:
            order = tuple(reversed(order))
        values = {}
        for name in order:
            arguments = {"status": status} if name == "read_status" else {}
            values[name] = await context.tools.call(name, **arguments)
        decision = "PROCEED" if values["read_status"] == values["read_rule"] else "REVIEW"
        return AgentResult(
            output_text=f"DECISION: {decision}",
            trajectory=context.tools.trajectory(),
            wire_request=WireRequest.from_payload(
                provider="local",
                model="status-policy",
                adapter="examples.dfah_gate_loop",
                adapter_version=version,
                payload={"model": "status-policy", "case_id": case.case_id, **parameters},
                parameters=parameters,
            ),
        )

    return candidate


def run_review_loop(out: Path) -> tuple[GateResult, ...]:
    """Evaluate the two supplied candidates and preserve all evidence for review."""

    out.mkdir(parents=True, exist_ok=False)
    (out / "policy.json").write_text(POLICY.model_dump_json(indent=2) + "\n")
    gate = Gate(POLICY)
    outcomes = []
    summary = []
    for name, version, vary_order in CANDIDATES:
        candidate = make_candidate(version, vary_order=vary_order)
        assert candidate.suite is not None
        run_dir = out / name
        Replay(
            suite=candidate.suite,
            replays=2,
            seed=42,
            out=run_dir,
            episode_timeout_s=5,
        ).run(candidate)
        # Reload from the committed store before making a gate decision.
        report = Report.from_json(run_dir)
        result = gate.evaluate(report)
        report.to_html(run_dir / "report.html")
        (run_dir / "gate.json").write_text(result.model_dump_json(indent=2) + "\n")
        failed = [check.name for check in result.checks if not check.passed]
        assert report.tar is not None
        summary.append(
            {
                "candidate": name,
                "adapter_version": version,
                "manifest_sha256": report.manifest.hash,
                "episode_root_sha256": report.episode_artifact_root_sha256,
                "passed": result.passed,
                "failed_checks": failed,
                "dar": report.dar,
                "tar_seq": report.tar.seq,
                "flagged_groups": report.flagged_groups,
                "report_html": f"{name}/report.html",
            }
        )
        outcomes.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"{name}: {status} DAR={report.dar:.3f} TARseq={report.tar.seq:.3f}")
        if failed:
            print(f"  Failed checks: {', '.join(failed)}")
        print(f"  Report: {run_dir / 'report.html'}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return tuple(outcomes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".dfah/review-loop"))
    args = parser.parse_args()
    try:
        outcomes = run_review_loop(args.out.resolve())
    except FileExistsError:
        parser.error("output directory already exists; choose a new --out to preserve it")
    # The first failure is intentional; the supplied correction must pass.
    raise SystemExit(0 if outcomes[-1].passed else 1)


if __name__ == "__main__":
    main()
