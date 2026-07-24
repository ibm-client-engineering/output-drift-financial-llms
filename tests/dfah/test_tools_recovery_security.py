from __future__ import annotations

import multiprocessing
import os
import socket
from datetime import datetime, timezone

import anyio
import pytest
from pydantic import ValidationError

from dfah import (
    AgentResult,
    ArtifactError,
    Case,
    ChannelState,
    ConfigurationError,
    Replay,
    Suite,
    ToolCall,
    ToolExecutionError,
    ToolExecutionState,
    ToolRegistry,
    ToolSpec,
    Trajectory,
    WireRequest,
    agent,
    build_manifest,
)
from dfah._canonical import canonical_bytes
from dfah.exceptions import EpisodeConflictError, OptionalDependencyError
from dfah.metrics.eligibility import evaluate_group
from dfah.store import DispatchIntent, EpisodeStart, FileStore, RunLease
from dfah.testing import check_agent


def _contend_for_stale_lease(root, ready, start, release, results):
    """Process target proving stale recovery still has exactly one live winner."""

    store = FileStore(root)
    ready.put("ready")
    start.wait()
    try:
        with store.run_lease(manifest_hash="a" * 64, recover_stale=True):
            results.put("acquired")
            release.wait()
    except EpisodeConflictError:
        results.put("blocked")
    except Exception as exc:  # pragma: no cover - returned for parent diagnostics
        results.put(f"error:{type(exc).__name__}:{exc}")
    finally:
        store.close()


def _agent_for(suite: Suite, calls: dict[str, int]):
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="fake-v1",
        adapter="tests.recovery",
        request_parameters=parameters,
        git_sha="f" * 40,
    )

    @agent(manifest=manifest, suite=suite)
    async def candidate(case, context):
        calls["count"] += 1
        return AgentResult(
            output_text=f"DECISION: {suite.decisions[0]}",
            trajectory=Trajectory(state=ChannelState.OBSERVED_EMPTY),
            wire_request=WireRequest.from_payload(
                provider="fake",
                model="fake-v1",
                payload={"case_id": case.case_id, **parameters},
                parameters=parameters,
                adapter="tests.recovery",
            ),
        )

    return candidate


def _start_for(agent_value, suite: Suite, case: Case, replay_index: int = 0):
    return EpisodeStart(
        manifest_hash=agent_value.manifest.hash,
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        case_id=case.case_id,
        task=case.task,
        replay_index=replay_index,
        started_at=datetime.now(timezone.utc),
        idempotency_key="a" * 64,
    )


def test_planned_only_episode_is_safe_to_resume(tmp_path):
    suite = Suite.load("compliance-v1")
    calls = {"count": 0}
    candidate = _agent_for(suite, calls)
    store = FileStore(tmp_path / "run")
    start = _start_for(candidate, suite, suite.cases[0])
    store.start(start)
    assert store.inspect(*start.key) == "started"

    report = Replay(suite=suite, replays=2, out=tmp_path / "run").run(candidate)
    assert report.status.value == "complete"
    assert calls["count"] == 4


def test_dispatch_boundary_is_never_silently_reissued(tmp_path):
    suite = Suite.load("compliance-v1")
    calls = {"count": 0}
    candidate = _agent_for(suite, calls)
    store = FileStore(tmp_path / "run")
    start = _start_for(candidate, suite, suite.cases[0])
    store.start(start)
    store.mark_dispatching(
        DispatchIntent(
            manifest_hash=start.manifest_hash,
            case_id=start.case_id,
            replay_index=start.replay_index,
            marked_at=datetime.now(timezone.utc),
        )
    )
    assert store.inspect(*start.key) == "dispatching"

    report = Replay(suite=suite, replays=2, out=tmp_path / "run").run(candidate)
    assert report.status.value == "partial"
    assert calls["count"] == 3
    episode = store.read(*start.key)
    assert episode.status.value == "interrupted"
    assert episode.dispatch_state.value == "unknown_after_dispatch"


def test_all_dangling_dispatch_reservations_settle_before_new_workers(tmp_path):
    suite = Suite.load("compliance-v1")
    calls = {"count": 0}
    candidate = _agent_for(suite, calls)
    store = FileStore(tmp_path / "multi-dangling")
    for case in suite.cases:
        start = _start_for(candidate, suite, case)
        store.start(start)
        store.mark_dispatching(
            DispatchIntent(
                manifest_hash=start.manifest_hash,
                case_id=start.case_id,
                replay_index=start.replay_index,
                marked_at=datetime.now(timezone.utc),
                reserved_cost_usd=0.001,
            )
        )

    report = Replay(
        suite=suite,
        replays=2,
        out=tmp_path / "multi-dangling",
        concurrency=4,
        budget_usd=0.002,
        estimated_max_episode_cost_usd=0.001,
    ).run(candidate)
    assert calls["count"] == 0
    assert report.total_cost_usd == pytest.approx(0.002)
    assert report.episodes_completed == 2
    assert report.status.value == "partial"


def test_outside_design_dispatch_is_visible_and_blocks_before_agent_call(tmp_path):
    suite = Suite.load("compliance-v1")
    calls = {"count": 0}
    candidate = _agent_for(suite, calls)
    run_dir = tmp_path / "outside-design"
    store = FileStore(run_dir)
    start = _start_for(candidate, suite, suite.cases[0], replay_index=2)
    store.start(start)
    store.mark_dispatching(
        DispatchIntent(
            manifest_hash=start.manifest_hash,
            case_id=start.case_id,
            replay_index=start.replay_index,
            marked_at=datetime.now(timezone.utc),
            reserved_cost_usd=0.001,
        )
    )

    with pytest.raises(EpisodeConflictError, match="outside the bound replay design"):
        Replay(suite=suite, replays=2, out=run_dir).run(candidate)
    assert calls["count"] == 0
    assert store.inspect(*start.key) == "dispatching"


def test_run_directory_allows_only_one_writer_and_preserves_stale_lease(tmp_path):
    store = FileStore(tmp_path / "leased")
    manifest_hash = "a" * 64
    with store.run_lease(manifest_hash=manifest_hash):
        with (
            pytest.raises(EpisodeConflictError, match="active writer"),
            store.run_lease(manifest_hash=manifest_hash),
        ):
            raise AssertionError("unreachable")
        with (
            pytest.raises(EpisodeConflictError, match="active writer"),
            store.run_lease(manifest_hash=manifest_hash, recover_stale=True),
        ):
            raise AssertionError("unreachable")
    assert not store.lease.exists()

    dead_pid = max(os.getpid() + 1_000_000, 99_999_999)
    stale = RunLease(
        token="b" * 32,
        pid=dead_pid,
        hostname=socket.gethostname(),
        manifest_hash=manifest_hash,
        acquired_at=datetime.now(timezone.utc),
    )
    payload = canonical_bytes(stale, redact=True) + b"\n"
    store._exclusive_write(store.lease, payload)
    with store.run_lease(manifest_hash=manifest_hash, recover_stale=True) as recovered:
        assert recovered.token != stale.token
        assert len(tuple(store.stale_leases.glob("*.json"))) == 1
    assert not store.lease.exists()


def test_two_processes_cannot_both_win_stale_lease_recovery(tmp_path):
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - package FileStore is POSIX-only
        pytest.skip("POSIX fork context is unavailable")
    run_dir = tmp_path / "stale-race"
    store = FileStore(run_dir)
    stale = RunLease(
        token="c" * 32,
        pid=max(os.getpid() + 1_000_000, 99_999_999),
        hostname=socket.gethostname(),
        manifest_hash="a" * 64,
        acquired_at=datetime.now(timezone.utc),
    )
    store._exclusive_write(store.lease, canonical_bytes(stale, redact=True) + b"\n")
    store.close()

    ready = context.Queue()
    results = context.Queue()
    start = context.Event()
    release = context.Event()
    processes = [
        context.Process(
            target=_contend_for_stale_lease,
            args=(run_dir, ready, start, release, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    assert [ready.get(timeout=5) for _ in processes] == ["ready", "ready"]
    start.set()
    outcomes = [results.get(timeout=5) for _ in processes]
    release.set()
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0
    assert sorted(outcomes) == ["acquired", "blocked"]


def test_store_rejects_writer_guard_replacement(tmp_path):
    store = FileStore(tmp_path / "guard-replacement")
    original = tmp_path / "original-writer-guard"
    store.lease_guard.rename(original)
    store.lease_guard.write_bytes(b"")

    with (
        pytest.raises(ArtifactError, match="guard identity changed"),
        store.run_lease(manifest_hash="a" * 64),
    ):
        raise AssertionError("unreachable")


def test_filestore_rejects_symlink_root_without_touching_target(tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o755)
    before_mode = victim.stat().st_mode
    link = tmp_path / "run-link"
    link.symlink_to(victim, target_is_directory=True)

    with pytest.raises(ArtifactError, match="run root is a symlink"):
        FileStore(link)
    assert victim.stat().st_mode == before_mode
    assert not (victim / "episodes").exists()


def test_replay_rejects_symlink_run_plan_before_agent_call(tmp_path):
    suite = Suite.load("compliance-v1")
    calls = {"count": 0}
    candidate = _agent_for(suite, calls)
    run_dir = tmp_path / "plan-link"
    store = FileStore(run_dir)
    external = tmp_path / "external-plan.json"
    external.write_text("{}", encoding="utf-8")
    store.plan.symlink_to(external)

    with pytest.raises(ArtifactError, match="unsafe"):
        Replay(suite=suite, replays=2, out=run_dir).run(candidate)
    assert calls["count"] == 0
    assert external.read_text(encoding="utf-8") == "{}"


def test_pinned_store_rejects_managed_directory_replacement(tmp_path):
    suite = Suite.load("compliance-v1")
    candidate = _agent_for(suite, {"count": 0})
    run_dir = tmp_path / "pinned-child"
    store = FileStore(run_dir)
    original = tmp_path / "original-starts"
    store.starts.rename(original)
    victim = tmp_path / "child-victim"
    victim.mkdir()
    store.starts.symlink_to(victim, target_is_directory=True)
    start = _start_for(candidate, suite, suite.cases[0])

    with pytest.raises(ArtifactError, match="directory identity changed"):
        store.start(start)
    assert not tuple(victim.iterdir())


def test_pinned_store_rejects_run_root_replacement(tmp_path):
    suite = Suite.load("compliance-v1")
    candidate = _agent_for(suite, {"count": 0})
    run_dir = tmp_path / "pinned-root"
    store = FileStore(run_dir)
    old_root = tmp_path / "pinned-root-old"
    run_dir.rename(old_root)
    victim = tmp_path / "root-victim"
    victim_store = FileStore(victim)
    victim_store.close()
    run_dir.symlink_to(victim, target_is_directory=True)
    start = _start_for(candidate, suite, suite.cases[0])

    with pytest.raises(ArtifactError, match="directory identity changed"):
        store.start(start)
    assert not tuple((victim / "starts").iterdir())


def test_tool_registry_records_hashes_without_raw_arguments(tmp_path):
    spec = ToolSpec(
        name="lookup_limit",
        input_schema={
            "type": "object",
            "properties": {"entity": {"type": "string"}},
            "required": ["entity"],
        },
    )
    suite = Suite(
        suite_id="tool-registry-test",
        suite_version="1.0.0",
        decisions=("pass", "review"),
        cases=(Case(case_id="C-1", input={"entity": "synthetic-1"}),),
        tools=(spec,),
    )
    registry = ToolRegistry()

    @registry.tool(spec)
    def lookup_limit(*, entity: str):
        return {"entity": entity, "limit": 10}

    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="fake-tool-v1",
        adapter="tests.tools",
        request_parameters=parameters,
        git_sha="f" * 40,
    )

    @agent(manifest=manifest, tools=registry, suite=suite)
    async def candidate(case, context):
        assert context.tools is not None
        await context.tools.call("lookup_limit", entity=case.input["entity"])
        return AgentResult(
            output_text="DECISION: PASS",
            trajectory=context.tools.trajectory(),
            wire_request=WireRequest.from_payload(
                provider="fake",
                model="fake-tool-v1",
                payload={"model": "fake-tool-v1", **parameters},
                parameters=parameters,
                adapter="tests.tools",
            ),
        )

    report = Replay(
        suite=suite,
        replays=2,
        out=tmp_path / "run",
    ).run(candidate)
    assert report.status.value == "complete"
    episodes = FileStore(tmp_path / "run").list(manifest_hash=manifest.hash)
    recorded = episodes[0].trajectory.tool_calls[0]
    assert recorded.arguments_redacted
    assert dict(recorded.arguments) == {}
    assert recorded.arguments_hash == recorded.argument_hash
    assert recorded.output_hash is not None
    assert check_agent(candidate).passed


@pytest.mark.parametrize(
    "arguments",
    [
        {"entity": 123, "options": {"limit": 1}},
        {"options": {"limit": 1}},
        {"entity": "synthetic", "options": {"limit": 1}, "extra": True},
        {"entity": "synthetic", "options": {"limit": "1"}},
    ],
    ids=["wrong-type", "missing-required", "extra-property", "nested-wrong-type"],
)
def test_tool_registry_rejects_schema_violations_before_execution(arguments):
    spec = ToolSpec(
        name="lookup",
        input_schema={
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "options": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer"}},
                    "required": ["limit"],
                    "additionalProperties": False,
                },
            },
            "required": ["entity", "options"],
            "additionalProperties": False,
        },
    )
    registry = ToolRegistry()
    invoked = {"count": 0}

    @registry.tool(spec)
    def lookup(*, entity: str, options: dict[str, int]):
        invoked["count"] += 1
        return {"entity": entity, "limit": options["limit"]}

    session = registry.session()

    async def invoke() -> None:
        with pytest.raises(ToolExecutionError, match="arguments violate its JSON Schema"):
            await session.call("lookup", **arguments)

    anyio.run(invoke)
    assert invoked["count"] == 0
    rejected = session.calls[-1]
    assert rejected.execution_state is ToolExecutionState.REJECTED
    assert rejected.result_state is ChannelState.MALFORMED
    assert dict(rejected.arguments) == {}


def test_tool_registry_rejects_invalid_schema_at_registration():
    with pytest.raises(ValidationError, match="not valid Draft 2020-12"):
        ToolSpec(name="broken", input_schema={"type": "not-a-json-type"})

    with pytest.raises(ValidationError, match="must declare type=object"):
        ToolSpec(name="broken", input_schema={})


def test_tool_registry_freezes_before_episode_sessions():
    spec = ToolSpec(name="first", input_schema={"type": "object"})
    registry = ToolRegistry()
    registry.register(spec, lambda: "ok")
    registry.session()

    with pytest.raises(ConfigurationError, match="frozen"):
        registry.register(
            ToolSpec(name="late", input_schema={"type": "object"}),
            lambda: "late",
        )


def test_conformance_accepts_a_stable_zero_tool_agent():
    suite = Suite.load("compliance-v1")
    candidate = _agent_for(suite, {"count": 0})
    result = check_agent(candidate)
    assert result.passed, result
    tool_check = next(
        check for check in result.checks if check.name == "deterministic_tool_outputs"
    )
    assert tool_check.skipped
    assert "not applicable" in tool_check.detail


def test_conformance_preflight_bounds_large_suites_to_two_cases():
    base = Suite.load("compliance-v1")
    suite = Suite(
        suite_id="bounded-conformance",
        suite_version="1.0.0",
        decisions=base.decisions,
        cases=tuple(
            Case(case_id=f"C-{index:03d}", task="smoke", input={"index": index})
            for index in range(25)
        ),
    )
    calls = {"count": 0}
    candidate = _agent_for(suite, calls)
    result = check_agent(candidate)
    assert result.passed
    assert result.cases_selected == 2
    assert result.episodes_planned == 4
    assert calls["count"] == 4


def test_conformance_warns_on_replay_visible_decision_variation():
    suite = Suite.load("compliance-v1")
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="ambient-state-v1",
        adapter="tests.ambient",
        request_parameters=parameters,
        git_sha="f" * 40,
    )

    @agent(manifest=manifest, suite=suite)
    async def candidate(case, context):
        label = "ESCALATE" if context.replay_index == 0 else "DISMISS"
        return AgentResult(
            output_text=f"DECISION: {label}",
            trajectory=Trajectory(state=ChannelState.OBSERVED_EMPTY),
            wire_request=WireRequest.from_payload(
                provider="fake",
                model="ambient-state-v1",
                payload={"case_id": case.case_id, **parameters},
                parameters=parameters,
                adapter="tests.ambient",
            ),
        )

    result = check_agent(candidate, raise_on_error=True)
    assert result.passed
    failed = {check.name for check in result.checks if not check.passed}
    assert failed == set()
    stability = next(check for check in result.checks if check.name == "replay_stability_smoke")
    assert not stability.skipped
    assert stability.detail.startswith("WARNING:")
    assert "observational model result" in stability.detail


def test_conformance_still_rejects_nondeterministic_declared_tool_outputs():
    spec = ToolSpec(
        name="lookup_limit",
        input_schema={
            "type": "object",
            "properties": {"entity": {"type": "string"}},
            "required": ["entity"],
            "additionalProperties": False,
        },
    )
    suite = Suite(
        suite_id="nondeterministic-tool-test",
        suite_version="1.0.0",
        decisions=("pass", "review"),
        cases=(Case(case_id="C-1", input={"entity": "synthetic-1"}),),
        tools=(spec,),
    )
    registry = ToolRegistry()
    invocations = {"count": 0}

    @registry.tool(spec)
    def lookup_limit(*, entity: str):
        invocations["count"] += 1
        return {"entity": entity, "invocation": invocations["count"]}

    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="nondeterministic-tool-v1",
        adapter="tests.nondeterministic_tool",
        request_parameters=parameters,
        git_sha="f" * 40,
    )

    @agent(manifest=manifest, tools=registry, suite=suite)
    async def candidate(case, context):
        assert context.tools is not None
        await context.tools.call("lookup_limit", entity=case.input["entity"])
        return AgentResult(
            output_text="DECISION: PASS",
            trajectory=context.tools.trajectory(),
            wire_request=WireRequest.from_payload(
                provider="fake",
                model="nondeterministic-tool-v1",
                payload={"model": "nondeterministic-tool-v1", **parameters},
                parameters=parameters,
                adapter="tests.nondeterministic_tool",
            ),
        )

    result = check_agent(candidate)
    assert not result.passed
    failed = {check.name for check in result.checks if not check.passed}
    assert failed == {"deterministic_tool_outputs"}
    tool_check = next(
        check for check in result.checks if check.name == "deterministic_tool_outputs"
    )
    assert "identical tool inputs changed output" in tool_check.detail


def test_conformance_distinguishes_missing_evidence_from_path_variation(tmp_path):
    suite = Suite.load("compliance-v1")
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="always-fails-v1",
        adapter="tests.always_fails",
        request_parameters=parameters,
        git_sha="f" * 40,
    )

    @agent(manifest=manifest, suite=suite)
    async def candidate(case, context):
        raise RuntimeError("provider unavailable")

    result = check_agent(candidate)
    checks = {check.name: check for check in result.checks}
    assert not result.passed
    assert checks["replay_stability_smoke"].skipped
    assert "no replay group was eligible" in checks["replay_stability_smoke"].detail
    assert "variation was observed" not in checks["replay_stability_smoke"].detail
    assert not checks["eligible_capture"].passed
    assert "provider_error=4" in checks["eligible_capture"].detail
    assert "RuntimeError=4" in checks["eligible_capture"].detail
    assert "inspect terminal episodes" not in checks["eligible_capture"].detail

    retained = tmp_path / "conformance-diagnostics"
    retained_result = check_agent(candidate, out=retained)
    retained_check = next(
        check for check in retained_result.checks if check.name == "eligible_capture"
    )
    assert retained.exists()
    assert str(retained.resolve()) in retained_check.detail


def test_conformance_configuration_failure_is_actionable():
    suite = Suite.load("compliance-v1")
    candidate = _agent_for(suite, {"count": 0})
    result = check_agent(candidate, suite="conformance-v1")
    execution = next(check for check in result.checks if check.name == "execution")
    assert not execution.passed
    assert "ConfigurationError" in execution.detail
    assert "agent manifest does not match suite" in execution.detail
    assert "inspect local diagnostics" not in execution.detail


def test_conformance_does_not_claim_nonexistent_retained_diagnostics(tmp_path):
    suite = Suite.load("compliance-v1")
    candidate = _agent_for(suite, {"count": 0})
    requested = tmp_path / "never-created"
    result = check_agent(candidate, suite="conformance-v1", out=requested)
    execution = next(check for check in result.checks if check.name == "execution")
    assert not requested.exists()
    assert "diagnostics retained" not in execution.detail


def test_persisted_json_containers_are_recursively_immutable():
    case = Case(case_id="C", input={"items": [{"amount": 1}]}, metadata={"tags": ["x"]})
    with pytest.raises(TypeError):
        case.input["items"][0]["amount"] = 2
    with pytest.raises(TypeError):
        case.metadata["tags"][0] = "y"


def test_credentials_are_rejected_before_artifact_creation():
    suite = Suite.load("compliance-v1")
    with pytest.raises(ValidationError, match="credential-shaped"):
        build_manifest(
            suite,
            provider="fake",
            model="fake",
            adapter="tests.fake",
            request_parameters={
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "api_key": "sk-proj-abcdefghijklmnopqrstuvwxyz",
            },
            git_sha="f" * 40,
        )


@pytest.mark.parametrize("label", ["needs-review", "needs review", "yes/no"])
def test_suite_rejects_labels_the_strict_parser_cannot_emit(label):
    with pytest.raises(ValidationError, match="parser-compatible"):
        Suite(
            suite_id="bad-ontology",
            suite_version="1.0.0",
            decisions=(label, "other"),
            cases=(Case(case_id="C", input={}),),
        )


def test_suite_rejects_unsupported_required_channels():
    with pytest.raises(ValidationError, match="require exactly decision and trajectory"):
        Suite(
            suite_id="evidence-not-modeled",
            suite_version="1.0.0",
            decisions=("pass", "review"),
            cases=(Case(case_id="C", input={}),),
            required_channels=("decision", "trajectory", "evidence"),
        )


@pytest.mark.parametrize(
    "tool_call",
    [
        ToolCall(
            name="unknown",
            arguments={"entity": "x"},
            execution_state=ToolExecutionState.REQUESTED,
        ),
        ToolCall(
            name="declared",
            arguments={"entity": 7},
            execution_state=ToolExecutionState.REQUESTED,
        ),
    ],
    ids=["unknown-tool", "wrong-argument-type"],
)
def test_manual_trajectory_cannot_bypass_suite_tool_contract(tmp_path, tool_call):
    spec = ToolSpec(
        name="declared",
        input_schema={
            "type": "object",
            "properties": {"entity": {"type": "string"}},
            "required": ["entity"],
            "additionalProperties": False,
        },
    )
    suite = Suite(
        suite_id="manual-capture-contract",
        suite_version="1.0.0",
        decisions=("pass", "review"),
        cases=(Case(case_id="C", input={}),),
        tools=(spec,),
    )
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="manual-v1",
        adapter="tests.manual",
        request_parameters=parameters,
        git_sha="f" * 40,
    )

    @agent(manifest=manifest, suite=suite)
    async def candidate(case, context):
        return AgentResult(
            output_text="DECISION: PASS",
            trajectory=Trajectory(
                state=ChannelState.OBSERVED_NONEMPTY, tool_calls=(tool_call,)
            ),
            wire_request=WireRequest.from_payload(
                provider="fake",
                model="manual-v1",
                payload={"case_id": case.case_id, **parameters},
                parameters=parameters,
                adapter="tests.manual",
            ),
        )

    report = Replay(suite=suite, replays=2, out=tmp_path / "manual").run(candidate)
    assert report.status.value == "partial"
    episodes = FileStore(tmp_path / "manual").list(manifest_hash=manifest.hash)
    assert all(episode.status.value == "contract_error" for episode in episodes)
    assert all(not episode.trajectory.tool_calls for episode in episodes)


def test_payload_variation_makes_an_otherwise_valid_group_ineligible(tmp_path):
    suite = Suite.load("compliance-v1")
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="nonce-v1",
        adapter="tests.nonce",
        request_parameters=parameters,
        endpoint="https://synthetic.invalid/v1",
        git_sha="f" * 40,
    )

    @agent(manifest=manifest, suite=suite)
    async def candidate(case, context):
        return AgentResult(
            output_text="DECISION: ESCALATE",
            trajectory=Trajectory(state=ChannelState.OBSERVED_EMPTY),
            wire_request=WireRequest.from_payload(
                provider="fake",
                model="nonce-v1",
                endpoint="https://synthetic.invalid/v1",
                payload={"case_id": case.case_id, "nonce": context.replay_index, **parameters},
                parameters=parameters,
                adapter="tests.nonce",
            ),
        )

    run_dir = tmp_path / "payload-mismatch"
    report = Replay(suite=suite, replays=2, out=run_dir).run(candidate)
    assert report.status.value == "partial"
    assert report.episodes_completed == 4
    assert report.episodes_eligible == 0
    episodes = FileStore(run_dir).list(manifest_hash=manifest.hash)
    first_group = [episode for episode in episodes if episode.case_id == suite.cases[0].case_id]
    eligibility = evaluate_group(first_group, required_replays=2)
    assert not eligibility.eligible
    assert "wire_payload_mismatch" in eligibility.reasons


def test_redacted_tool_call_cannot_retain_raw_arguments():
    with pytest.raises(ValidationError, match="must not retain raw"):
        ToolCall(
            name="lookup",
            arguments={"account": "synthetic-123"},
            arguments_hash="a" * 64,
            arguments_redacted=True,
        )


def test_executed_tool_call_requires_an_observed_hashed_result():
    with pytest.raises(ValidationError, match="requires an observed result channel"):
        ToolCall(name="lookup", execution_state=ToolExecutionState.EXECUTED)
    with pytest.raises(ValidationError, match="requires an output hash"):
        ToolCall(
            name="lookup",
            execution_state=ToolExecutionState.EXECUTED,
            result_state=ChannelState.OBSERVED_EMPTY,
        )


def test_wire_parameters_are_attested_to_top_level_or_nested_payloads():
    with pytest.raises(ValueError, match="differs from wire payload"):
        WireRequest.from_payload(
            provider="fake",
            model="fake-v1",
            payload={"settings": {"temperature": 0.7}},
            parameters={"temperature": 0.0},
            parameter_paths={"temperature": ("settings", "temperature")},
            adapter="tests.attestation",
        )

    request = WireRequest.from_payload(
        provider="fake",
        model="fake-v1",
        payload={"settings": {"temperature": 0.0}},
        parameters={"temperature": 0.0},
        parameter_paths={"temperature": ("settings", "temperature")},
        adapter="tests.attestation",
    )
    assert request.parameters_attested


def test_adapter_version_is_not_cryptographic_source_provenance(monkeypatch):
    suite = Suite.load("compliance-v1")
    monkeypatch.setattr("dfah.replay._git_sha", lambda: None)
    with pytest.raises(ValidationError, match="implementation hash"):
        build_manifest(
            suite,
            provider="fake",
            model="fake-v1",
            adapter="tests.provenance",
            adapter_version="1.2.3",
            request_parameters={"temperature": 0.0},
        )

    manifest = build_manifest(
        suite,
        provider="fake",
        model="fake-v1",
        adapter="tests.provenance",
        adapter_version="1.2.3",
        implementation_hash="d" * 64,
        request_parameters={"temperature": 0.0},
    )
    assert manifest.implementation_hash == "d" * 64


def test_artifact_case_id_keeps_raw_case_identity_out_of_run_artifacts(tmp_path):
    raw_case_id = "customer-visible-reference-123"
    artifact_case_id = "case-7f29c1"
    suite = Suite(
        suite_id="pseudonymized-cases",
        suite_version="1.0.0",
        decisions=("pass", "review"),
        cases=(
            Case(
                case_id=raw_case_id,
                artifact_case_id=artifact_case_id,
                input={"synthetic": True},
            ),
        ),
    )
    parameters = {"temperature": 0.0}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="fake-v1",
        adapter="tests.pseudonym",
        request_parameters=parameters,
        git_sha="f" * 40,
    )

    @agent(manifest=manifest, suite=suite)
    async def candidate(case, context):
        assert case.case_id == raw_case_id
        assert context.case_id == artifact_case_id
        return AgentResult(
            output_text="DECISION: PASS",
            trajectory=Trajectory(state=ChannelState.OBSERVED_EMPTY),
            wire_request=WireRequest.from_payload(
                provider="fake",
                model="fake-v1",
                payload={"case_id": case.case_id, **parameters},
                parameters=parameters,
                adapter="tests.pseudonym",
            ),
        )

    run_dir = tmp_path / "pseudonymized"
    report = Replay(suite=suite, replays=2, out=run_dir).run(candidate)
    assert {row.case_id for row in report.case_reports} == {artifact_case_id}
    artifact_bytes = b"\n".join(
        path.read_bytes() for path in run_dir.rglob("*") if path.is_file()
    )
    assert raw_case_id.encode() not in artifact_bytes


def test_structural_agent_wrong_result_type_commits_contract_error(tmp_path):
    suite = Suite.load("compliance-v1")
    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="bad-contract-v1",
        adapter="tests.bad_contract",
        request_parameters=parameters,
        git_sha="f" * 40,
    )
    calls = {"count": 0}

    class BadAgent:
        async def arun(self, case, context):
            calls["count"] += 1
            return {"decision": "pass"}

        @property
        def manifest(self):
            return manifest

    run_dir = tmp_path / "bad-result"
    report = Replay(
        suite=suite,
        replays=2,
        out=run_dir,
        budget_usd=0.001,
        estimated_max_episode_cost_usd=0.001,
    ).run(BadAgent())
    assert calls["count"] == 1
    assert report.status.value == "partial"
    episode = FileStore(run_dir).list(manifest_hash=manifest.hash)[0]
    assert episode.status.value == "contract_error"
    assert episode.cost_usd == 0.001


def test_otel_preflight_fails_before_artifact_or_dispatch(tmp_path, monkeypatch):
    import dfah.replay as replay_module

    suite = Suite.load("compliance-v1")
    candidate = _agent_for(suite, {"count": 0})
    run_dir = tmp_path / "otel-preflight"

    def fail_preflight(*, enabled: bool = False) -> None:
        assert enabled
        raise OptionalDependencyError("synthetic missing OTel")

    monkeypatch.setattr(replay_module, "telemetry_preflight", fail_preflight)
    with pytest.raises(OptionalDependencyError, match="missing OTel"):
        Replay(suite=suite, replays=2, out=run_dir, otel=True).run(candidate)
    assert not run_dir.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_budget_and_cost_values_fail_closed(tmp_path, value):
    suite = Suite.load("compliance-v1")
    candidate = _agent_for(suite, {"count": 0})
    with pytest.raises(ConfigurationError):
        Replay(
            suite=suite,
            replays=2,
            out=tmp_path / "nonfinite",
            budget_usd=value,
            estimated_max_episode_cost_usd=0.01,
        ).run(candidate)
    with pytest.raises(ValidationError):
        AgentResult(
            output_text="DECISION: PASS",
            trajectory=Trajectory(state=ChannelState.OBSERVED_EMPTY),
            wire_request=WireRequest.from_payload(
                provider="fake",
                model="fake-v1",
                payload={"model": "fake-v1"},
                parameters={},
                adapter="tests.fake",
            ),
            cost_usd=value,
        )


def test_otel_exports_sanitized_error_spans_without_exception_events(tmp_path, monkeypatch):
    telemetry_module = pytest.importorskip("dfah.telemetry")
    trace_sdk = pytest.importorskip("opentelemetry.sdk.trace")
    export_sdk = pytest.importorskip("opentelemetry.sdk.trace.export")
    memory_sdk = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")

    provider = trace_sdk.TracerProvider()
    exporter = memory_sdk.InMemorySpanExporter()
    provider.add_span_processor(export_sdk.SimpleSpanProcessor(exporter))

    def local_tracer(enabled: bool):
        return provider.get_tracer("dfah-test") if enabled else None

    monkeypatch.setattr(telemetry_module, "_tracer", local_tracer)

    spec = ToolSpec(
        name="explode",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
    )
    suite = Suite(
        suite_id="otel-error-test",
        suite_version="1.0.0",
        decisions=("pass", "review"),
        cases=(Case(case_id="C-1", input={}),),
        tools=(spec,),
    )
    registry = ToolRegistry()

    @registry.tool(spec)
    def explode(*, message: str):
        raise RuntimeError(f"sensitive failure: {message}")

    parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
    manifest = build_manifest(
        suite,
        provider="fake",
        model="otel-v1",
        adapter="tests.otel",
        request_parameters=parameters,
        git_sha="f" * 40,
    )

    @agent(manifest=manifest, tools=registry, suite=suite)
    async def candidate(case, context):
        assert context.tools is not None
        await context.tools.call("explode", message="sk-proj-syntheticcredential000000000")
        raise AssertionError("unreachable")

    Replay(suite=suite, replays=2, out=tmp_path / "otel", otel=True).run(candidate)
    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    assert all(not span.events for span in spans)
    span_attributes = [dict(span.attributes or {}) for span in spans]
    assert {attributes.get("gen_ai.operation.name") for attributes in span_attributes} == {
        "execute_tool",
        "invoke_agent",
    }
    tool_attributes = [
        attributes
        for attributes in span_attributes
        if attributes.get("gen_ai.operation.name") == "execute_tool"
    ]
    assert tool_attributes
    assert all(
        attributes.get("gen_ai.tool.type") == "function" for attributes in tool_attributes
    )
    assert all(
        "dfah.tool.arguments.sha256" not in attributes
        and "dfah.tool.result.sha256" not in attributes
        for attributes in span_attributes
    )
    serialized = "\n".join(
        f"{span.name} {dict(span.attributes or {})} {span.status.status_code}" for span in spans
    )
    assert "syntheticcredential" not in serialized
    assert "sensitive failure" not in serialized
    assert all((span.attributes or {}).get("error.type") for span in spans)
