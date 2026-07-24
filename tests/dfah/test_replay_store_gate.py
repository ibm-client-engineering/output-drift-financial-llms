from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anyio
import pytest
from pydantic import ValidationError

from dfah import (
    AgentResult,
    ArtifactError,
    ChannelState,
    Gate,
    GatePolicy,
    Replay,
    Report,
    Suite,
    TaskGatePolicy,
    ToolCall,
    ToolSpec,
    Trajectory,
    Usage,
    WireRequest,
    agent,
    build_manifest,
)
from dfah.exceptions import ConfigurationError, EpisodeConflictError
from dfah.report import render_markdown
from dfah.store import FileStore
from dfah.testing import check_agent


def make_agent(tmp_path: Path):
    base = Suite.load("compliance-v1")
    argument_schema = {
        "type": "object",
        "properties": {"case_id": {"type": "string"}},
        "required": ["case_id"],
        "additionalProperties": False,
    }
    suite = Suite(
        suite_id="test-tool-paths",
        suite_version="1.0.0",
        decisions=base.decisions,
        cases=base.cases,
        tools=(
            ToolSpec(name="check_control", input_schema=argument_schema),
            ToolSpec(name="get_profile", input_schema=argument_schema),
        ),
    )
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="fake-tool-agent-v1",
        adapter="tests.fake",
        request_parameters=parameters,
        git_sha="f" * 40,
    )
    calls = {"count": 0}

    @agent(manifest=manifest, suite=suite)
    async def fake(case, context):
        calls["count"] += 1
        check = ToolCall(
            name="check_control",
            arguments={"case_id": case.case_id},
            output_hash="1" * 64,
            result_state=ChannelState.OBSERVED_NONEMPTY,
        )
        profile = ToolCall(
            name="get_profile",
            arguments={"case_id": case.case_id},
            output_hash="2" * 64,
            result_state=ChannelState.OBSERVED_NONEMPTY,
        )
        ordered = (check, profile) if context.replay_index == 0 else (profile, check)
        payload = {
            "model": "fake-tool-agent-v1",
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        }
        return AgentResult(
            output_text="DECISION: ESCALATE",
            trajectory=Trajectory(state=ChannelState.OBSERVED_NONEMPTY, tool_calls=ordered),
            wire_request=WireRequest.from_payload(
                provider="fake",
                model="fake-tool-agent-v1",
                payload=payload,
                parameters=parameters,
                adapter="tests.fake",
            ),
            usage=Usage(input_tokens=10, output_tokens=2),
            cost_usd=0.001,
        )

    return fake, calls


def make_wire_unstable_agent():
    suite = Suite.load("compliance-v1")
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="wire-unstable-v1",
        adapter="tests.wire-unstable",
        request_parameters=parameters,
        git_sha="f" * 40,
    )

    @agent(manifest=manifest, suite=suite)
    async def candidate(case, context):
        payload = {
            "model": "wire-unstable-v1",
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
                adapter="tests.wire-unstable",
            ),
        )

    return candidate


def test_replay_golden_path_is_complete_and_idempotent(tmp_path):
    candidate, calls = make_agent(tmp_path)
    run_dir = tmp_path / "run"
    replay = Replay(suite=candidate.suite, replays=2, seed=42, out=run_dir)
    report = replay.run(candidate)
    assert report.status.value == "complete"
    assert report.dar == 1.0
    assert report.tar.seq == 0.5
    assert report.tar.set == 1.0
    assert report.flags_per_100_cases == 100.0
    assert all(
        row.eligibility.evidence is ChannelState.UNAVAILABLE for row in report.case_reports
    )
    assert calls["count"] == 4
    first_episode_bytes = {
        path.name: path.read_bytes() for path in (run_dir / "episodes").glob("*.json")
    }

    second = replay.run(candidate)
    assert calls["count"] == 4
    assert second.dar == report.dar and second.tar == report.tar
    assert first_episode_bytes == {
        path.name: path.read_bytes() for path in (run_dir / "episodes").glob("*.json")
    }
    assert len(list((run_dir / "reports").glob("*.json"))) == 2


def test_zero_eligible_groups_use_unavailable_metrics_and_reason_counts(tmp_path):
    candidate = make_wire_unstable_agent()
    run_dir = tmp_path / "ineligible"
    report = Replay(suite=candidate.suite, replays=2, out=run_dir).run(candidate)

    assert report.status.value == "partial"
    assert report.observed_groups == 0
    assert report.ineligible_groups == 2
    assert report.episodes_eligible == 0
    assert report.dar is None
    assert report.tar is None
    assert report.gap is None
    assert report.flags_per_100_cases is None
    assert report.sequence_flags_per_100_cases is None
    assert [(reason.reason, reason.groups) for reason in report.ineligibility_reasons] == [
        ("wire_payload_mismatch", 2)
    ]

    serialized = json.loads(report.to_json())
    assert serialized["dar"] is None
    assert serialized["tar"] is None
    assert serialized["gap"] is None
    zero_sentinel = report.model_dump(mode="json")
    zero_sentinel.update(
        {
            "dar": 0.0,
            "tar": {
                "schema_version": "1.0",
                "seq": 0.0,
                "bag": 0.0,
                "set": 0.0,
                "strong": 0.0,
            },
            "gap": 0.0,
        }
    )
    with pytest.raises(ValidationError, match="unavailable aggregate metrics"):
        Report.model_validate(zero_sentinel)
    assert "DAR: —" in render_markdown(report)
    assert "wire_payload_mismatch=2" in render_markdown(report)

    html = tmp_path / "ineligible.html"
    report.to_html(html)
    rendered = html.read_text(encoding="utf-8")
    assert "<strong>DAR</strong><br>—" in rendered
    assert "wire_payload_mismatch=2" in rendered

    permissive = Gate(
        GatePolicy(
            min_dar=0.0,
            min_tar_seq=0.0,
            max_gap=0.0,
            max_flags_per_100_cases=100.0,
            require_complete=False,
            require_artifact_verification=False,
        )
    ).evaluate(report)
    assert not permissive.passed
    assert {check.name for check in permissive.checks if not check.passed} == {
        "dar",
        "tar_seq",
        "gap",
        "flags_per_100_cases",
    }

    with pytest.raises(Exception, match="available aggregate metrics"):
        report.compare(report, allow_partial=True, allow_unverified=True)

    report_path = next((run_dir / "reports").glob("*.json"))
    tampered = json.loads(report_path.read_text(encoding="utf-8"))
    tampered["ineligibility_reasons"][0]["groups"] = 1
    report_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ArtifactError, match="eligibility diagnostics"):
        Report.from_json(run_dir)


def test_default_output_path_is_contract_and_design_specific(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    candidate, calls = make_agent(tmp_path)
    replay = Replay(suite=candidate.suite, replays=2, seed=42)
    run_dir = replay.output_path(candidate.manifest)
    assert run_dir.parent == Path(".dfah/runs")
    assert run_dir.name.startswith("test-tool-paths-")
    assert run_dir.name != "latest"

    replay.run(candidate)
    replay.run(candidate)
    assert calls["count"] == 4
    assert run_dir.is_dir()

    changed_design = Replay(suite=candidate.suite, replays=2, seed=43)
    assert changed_design.output_path(candidate.manifest) != run_dir


def test_gate_prices_flag_volume_and_cost(tmp_path):
    candidate, _ = make_agent(tmp_path)
    report = Replay(suite=candidate.suite, replays=2, out=tmp_path / "run").run(candidate)
    result = Gate(GatePolicy(max_flags_per_100_cases=5, max_cost_per_case_usd=0.01)).evaluate(
        report
    )
    assert not result.passed
    assert {check.name for check in result.checks if not check.passed} == {
        "flags_per_100_cases"
    }

    forged = report.model_copy(update={"dar": 0.123, "tar": report.tar})
    assert not forged.artifacts_verified
    forged_result = Gate(GatePolicy(min_dar=0.0)).evaluate(forged)
    assert not forged_result.passed
    assert {check.name for check in forged_result.checks if not check.passed} == {
        "artifact_verification"
    }


def test_gate_can_hold_each_task_to_its_own_threshold(tmp_path):
    candidate, _ = make_agent(tmp_path)
    report = Replay(suite=candidate.suite, replays=2, out=tmp_path / "task-gate").run(candidate)
    result = Gate(
        GatePolicy(
            require_complete=True,
            by_task=(
                TaskGatePolicy(task="compliance", min_tar_seq=0.9),
                TaskGatePolicy(task="missing-task", min_dar=0.9),
            ),
        )
    ).evaluate(report)
    assert not result.passed
    failures = {check.name for check in result.checks if not check.passed}
    assert failures == {
        "task[compliance].tar_seq",
        "task[missing-task].observed_groups",
        "task[missing-task].dar",
    }


def test_conformance_warns_on_replay_visible_path_variation(tmp_path):
    candidate, _ = make_agent(tmp_path)
    result = check_agent(candidate)
    assert result.passed
    failed = {check.name for check in result.checks if not check.passed}
    assert failed == set()
    stability = next(check for check in result.checks if check.name == "replay_stability_smoke")
    assert not stability.skipped
    assert stability.detail.startswith("WARNING:")
    assert "observational model result" in stability.detail
    assert {check.name for check in result.checks} >= {
        "deterministic_tool_outputs",
        "replay_stability_smoke",
        "parse_provenance",
        "wire_manifest_echo",
        "resumability_idempotence",
    }


def test_report_round_trip(tmp_path):
    candidate, _ = make_agent(tmp_path)
    report = Replay(suite=candidate.suite, replays=2, out=tmp_path / "run").run(candidate)
    path = tmp_path / "public-report.json"
    report.to_json(path)
    with pytest.raises(ArtifactError, match="standalone report"):
        Report.from_json(path)
    loaded = Report.from_json(path, allow_unverified=True)
    assert loaded.model_dump() == report.model_dump()
    assert not loaded.artifacts_verified
    unverified_gate = Gate(GatePolicy()).evaluate(loaded)
    assert not unverified_gate.passed
    assert {check.name for check in unverified_gate.checks if not check.passed} == {
        "artifact_verification"
    }
    verified = Report.from_json(tmp_path / "run")
    assert verified.artifacts_verified
    assert verified.model_dump() == report.model_dump()
    html = tmp_path / "report.html"
    report.to_html(html)
    assert "Flags / 100" in html.read_text()
    explanation = verified.explain_case(tmp_path / "run", candidate.suite.cases[0].case_id)
    assert len(explanation.episodes) == 2
    assert explanation.episodes[0].tool_sequence != explanation.episodes[1].tool_sequence
    assert all(call.arguments_hash for row in explanation.episodes for call in row.tool_calls)


def test_budget_stop_is_pre_dispatch_and_resumes_after_cap_increase(tmp_path):
    candidate, calls = make_agent(tmp_path)
    run_dir = tmp_path / "budgeted"
    partial = Replay(
        suite=candidate.suite,
        replays=2,
        out=run_dir,
        budget_usd=0.002,
        estimated_max_episode_cost_usd=0.001,
    ).run(candidate)
    assert partial.status.value == "partial"
    assert calls["count"] == 2
    assert len(FileStore(run_dir).list(manifest_hash=candidate.manifest.hash)) == 2

    complete = Replay(
        suite=candidate.suite,
        replays=2,
        out=run_dir,
        budget_usd=0.004,
        estimated_max_episode_cost_usd=0.001,
    ).run(candidate)
    assert complete.status.value == "complete"
    assert calls["count"] == 4
    assert complete.total_cost_usd == 0.004


def test_partial_terminal_commit_is_repaired_without_provider_reissue(tmp_path):
    candidate, calls = make_agent(tmp_path)
    run_dir = tmp_path / "repair"
    replay = Replay(suite=candidate.suite, replays=2, out=run_dir)
    report = replay.run(candidate)
    assert report.status.value == "complete" and calls["count"] == 4

    store = FileStore(run_dir)
    episode = store.list(manifest_hash=candidate.manifest.hash)[0]
    _, _, _, sidecar, commit = store._paths(*episode.key)
    sidecar.unlink()
    commit.unlink()
    assert store.inspect(*episode.key) == "corrupt"

    recovered = replay.run(candidate)
    assert recovered.status.value == "complete"
    assert calls["count"] == 4
    assert store.inspect(*episode.key) == "committed"


def test_underestimated_episode_cost_marks_budget_overshoot(tmp_path):
    candidate, calls = make_agent(tmp_path)
    report = Replay(
        suite=candidate.suite,
        replays=2,
        out=tmp_path / "underestimated",
        budget_usd=0.0005,
        estimated_max_episode_cost_usd=0.0005,
    ).run(candidate)
    assert calls["count"] == 1
    assert report.budget_exceeded
    assert report.status.value == "partial"
    assert "reservation bound" in (report.partial_reason or "")


def test_unknown_post_dispatch_cost_consumes_durable_reservation(tmp_path):
    suite = Suite.load("compliance-v1")
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="failing-provider-v1",
        adapter="tests.failing",
        request_parameters=parameters,
        git_sha="f" * 40,
    )
    calls = {"count": 0}

    @agent(manifest=manifest)
    async def failing(case, context):
        calls["count"] += 1
        raise RuntimeError("provider response was not captured")

    run_dir = tmp_path / "unknown-cost"
    replay = Replay(
        suite=suite,
        replays=2,
        out=run_dir,
        budget_usd=0.01,
        estimated_max_episode_cost_usd=0.01,
    )
    report = replay.run(failing)
    assert calls["count"] == 1
    assert report.status.value == "partial"
    assert report.total_cost_usd == 0.01
    assert report.episodes_completed == 1
    episode = FileStore(run_dir).list(manifest_hash=manifest.hash)[0]
    assert episode.status.value == "provider_error"
    dispatch = FileStore(run_dir).read_dispatch(*episode.key)
    assert dispatch.reserved_cost_usd == 0.01

    resumed = replay.run(failing)
    assert calls["count"] == 1
    assert resumed.total_cost_usd == 0.01


def test_recovered_terminal_cost_is_seeded_before_fresh_admission(tmp_path):
    suite = Suite.load("compliance-v1")
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="recover-cost-v1",
        adapter="tests.recover_cost",
        request_parameters=parameters,
        git_sha="f" * 40,
    )
    state = {"fail": True, "calls": 0}

    @agent(manifest=manifest)
    async def candidate(case, context):
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("synthetic post-dispatch failure")
        return AgentResult(
            output_text="DECISION: ESCALATE",
            trajectory=Trajectory(state=ChannelState.OBSERVED_EMPTY),
            wire_request=WireRequest.from_payload(
                provider="fake",
                model="recover-cost-v1",
                payload={"case_id": case.case_id, **parameters},
                parameters=parameters,
                adapter="tests.recover_cost",
            ),
            cost_usd=0.01,
        )

    run_dir = tmp_path / "recovered-cost"
    first = Replay(
        suite=suite,
        replays=2,
        out=run_dir,
        budget_usd=0.01,
        estimated_max_episode_cost_usd=0.01,
    ).run(candidate)
    assert first.total_cost_usd == 0.01 and state["calls"] == 1
    store = FileStore(run_dir)
    prior = store.list(manifest_hash=manifest.hash)[0]
    *_, commit = store._paths(*prior.key)
    commit.unlink()
    assert store.inspect(*prior.key) == "corrupt"

    state["fail"] = False
    resumed = Replay(
        suite=suite,
        replays=2,
        out=run_dir,
        budget_usd=0.03,
        estimated_max_episode_cost_usd=0.01,
    ).run(candidate)
    assert state["calls"] == 3
    assert resumed.total_cost_usd == pytest.approx(0.03)
    assert resumed.episodes_completed == 3
    assert resumed.status.value == "partial"
    assert not resumed.budget_exceeded


def test_report_rejects_impossible_aggregates_and_detects_artifact_tampering(tmp_path):
    candidate, _ = make_agent(tmp_path)
    run_dir = tmp_path / "integrity"
    report = Replay(suite=candidate.suite, replays=2, out=run_dir).run(candidate)
    impossible = report.model_dump(mode="json")
    impossible.update(
        {
            "episodes_planned": 999,
            "episodes_completed": 0,
            "episodes_eligible": 999,
            "observed_groups": 0,
            "flagged_groups": 999,
        }
    )
    with pytest.raises(ValidationError):
        Report.model_validate(impossible)

    episode_path = next((run_dir / "episodes").glob("*.json"))
    episode_path.write_bytes(episode_path.read_bytes() + b" ")
    with pytest.raises(ArtifactError, match="SHA-256"):
        Report.from_json(run_dir)


def test_report_artifact_verification_rejects_semantic_episode_rewrite(tmp_path):
    candidate, _ = make_agent(tmp_path)
    run_dir = tmp_path / "semantic-tamper"
    Replay(suite=candidate.suite, replays=2, out=run_dir).run(candidate)

    episode_path = next((run_dir / "episodes").glob("*.json"))
    episode_data = json.loads(episode_path.read_text(encoding="utf-8"))
    episode_data["suite_id"] = "tampered-suite"
    payload = (
        json.dumps(episode_data, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    )
    episode_path.write_bytes(payload)
    episode_path.with_suffix(".json.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
    )

    root, count = FileStore(run_dir).commitment(manifest_hash=candidate.manifest.hash)
    report_path = next((run_dir / "reports").glob("*.json"))
    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    report_data["episode_artifact_root_sha256"] = root
    report_data["episode_artifact_count"] = count
    report_path.write_text(json.dumps(report_data), encoding="utf-8")

    with pytest.raises(ArtifactError, match="metadata differs"):
        Report.from_json(run_dir)


def test_report_comparison_overrides_only_execution_manifest_differences(tmp_path):
    candidate, _ = make_agent(tmp_path)
    report = Replay(suite=candidate.suite, replays=2, out=tmp_path / "compare").run(candidate)

    execution_data = report.model_dump(mode="json")
    execution_data["manifest"]["provider"] = "another-provider"
    execution_data["manifest"]["model"] = "another-model"
    execution = Report.model_validate(execution_data)
    report.compare(execution, allow_cross_manifest=True, allow_unverified=True)

    changed_design = execution.model_copy(update={"schedule_seed": 99})
    with pytest.raises(Exception, match="populations or designs differ"):
        report.compare(changed_design, allow_cross_manifest=True, allow_unverified=True)

    fixture_data = report.model_dump(mode="json")
    fixture_data["manifest"]["fixture_hash"] = "b" * 64
    changed_fixture = Report.model_validate(fixture_data)
    with pytest.raises(Exception, match="fixtures, tools, or ontology"):
        report.compare(changed_fixture, allow_cross_manifest=True, allow_unverified=True)

    suite_data = report.model_dump(mode="json")
    suite_data["suite_id"] = "different-suite"
    suite_data["manifest"]["suite_id"] = "different-suite"
    changed_suite = Report.model_validate(suite_data)
    with pytest.raises(Exception, match="suite IDs differ"):
        report.compare(
            changed_suite,
            allow_cross_manifest=True,
            allow_cross_version=True,
            allow_unverified=True,
        )

    version_data = report.model_dump(mode="json")
    version_data["suite_version"] = "2.0.0"
    version_data["manifest"]["suite_version"] = "2.0.0"
    version_data["manifest"]["fixture_hash"] = "c" * 64
    changed_version = Report.model_validate(version_data)
    report.compare(changed_version, allow_cross_version=True, allow_unverified=True)

    with pytest.raises(Exception, match="artifact-verified"):
        report.compare(execution, allow_cross_manifest=True)

    partial = execution.model_copy(
        update={"status": "partial", "partial_reason": "synthetic incomplete report"}
    )
    with pytest.raises(Exception, match="complete reports"):
        report.compare(partial, allow_cross_manifest=True, allow_unverified=True)

    first_case = execution.case_reports[:1]
    unmatched = partial.model_copy(
        update={
            "case_reports": first_case,
            "observed_groups": 1,
            "episodes_eligible": first_case[0].replay_count,
        }
    )
    with pytest.raises(Exception, match="populations or designs differ"):
        report.compare(
            unmatched,
            allow_cross_manifest=True,
            allow_partial=True,
            allow_unverified=True,
        )


def test_run_directory_is_immutably_bound_to_one_replay_design(tmp_path):
    candidate, calls = make_agent(tmp_path)
    run_dir = tmp_path / "bound-plan"
    Replay(suite=candidate.suite, replays=2, seed=42, out=run_dir).run(candidate)
    assert calls["count"] == 4

    with pytest.raises(EpisodeConflictError, match="different replay design"):
        Replay(suite=candidate.suite, replays=3, seed=42, out=run_dir).run(candidate)
    assert calls["count"] == 4


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_episode_timeout_must_be_finite_and_positive(timeout):
    with pytest.raises(ConfigurationError, match="episode_timeout_s"):
        Replay(suite="compliance-v1", replays=2, episode_timeout_s=timeout)


def test_episode_timeout_is_unknown_after_dispatch_and_never_reissued(tmp_path):
    suite = Suite.load("compliance-v1")
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="slow-provider-v1",
        adapter="tests.slow",
        request_parameters=parameters,
        git_sha="f" * 40,
    )
    calls = {"count": 0}

    @agent(manifest=manifest)
    async def slow(case, context):
        calls["count"] += 1
        await anyio.sleep(1)
        raise AssertionError("the bounded adapter call should have timed out")

    run_dir = tmp_path / "bounded"
    replay = Replay(
        suite=suite,
        replays=2,
        out=run_dir,
        episode_timeout_s=0.01,
        estimated_max_episode_cost_usd=0.02,
    )
    report = replay.run(slow)
    assert calls["count"] == 4
    assert report.status.value == "partial"
    assert report.total_cost_usd == pytest.approx(0.08)

    store = FileStore(run_dir)
    assert store.read_plan().episode_timeout_s == 0.01
    episodes = store.list(manifest_hash=manifest.hash)
    assert len(episodes) == 4
    assert all(episode.status.value == "provider_error" for episode in episodes)
    assert all(episode.dispatch_state.value == "unknown_after_dispatch" for episode in episodes)
    assert all(episode.cost_usd == 0.02 for episode in episodes)
    assert all(
        episode.error is not None and episode.error.kind == "episode_timeout"
        for episode in episodes
    )

    resumed = replay.run(slow)
    assert calls["count"] == 4
    assert resumed.total_cost_usd == pytest.approx(0.08)

    with pytest.raises(EpisodeConflictError, match="different replay design"):
        Replay(
            suite=suite,
            replays=2,
            out=run_dir,
            episode_timeout_s=0.02,
            estimated_max_episode_cost_usd=0.02,
        ).run(slow)
    assert calls["count"] == 4


def test_gate_fails_undefined_group_denominators(tmp_path):
    suite = Suite.load("compliance-v1")
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="always-fails-v1",
        adapter="tests.failing",
        request_parameters=parameters,
        git_sha="f" * 40,
    )

    @agent(manifest=manifest)
    async def failing(case, context):
        raise RuntimeError("synthetic provider failure")

    report = Replay(
        suite=suite,
        replays=2,
        out=tmp_path / "zero-groups",
        budget_usd=0.001,
        estimated_max_episode_cost_usd=0.001,
    ).run(failing)
    result = Gate(
        GatePolicy(
            min_dar=0.0,
            min_tar_seq=0.0,
            max_gap=1.0,
            max_flags_per_100_cases=100.0,
            max_cost_per_case_usd=100.0,
            require_complete=False,
        )
    ).evaluate(report)
    assert not result.passed
    assert {check.name for check in result.checks if not check.passed} >= {
        "dar",
        "tar_seq",
        "gap",
        "flags_per_100_cases",
        "cost_per_case_usd",
    }
