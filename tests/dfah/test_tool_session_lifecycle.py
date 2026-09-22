"""An injected tool capability must not outlive its episode's capture boundary."""

import asyncio

import anyio
import pytest

from dfah import (
    AgentResult,
    Case,
    GatePolicy,
    Replay,
    Suite,
    ToolRegistry,
    ToolSpec,
    WireRequest,
    agent,
    build_manifest,
)
from dfah.exceptions import ToolExecutionError
from dfah.metrics import execution_summary
from dfah.store import FileStore


def lifecycle_agent(behavior):
    spec = ToolSpec(
        name="increment_counter",
        input_schema={"type": "object", "additionalProperties": False},
    )
    suite = Suite(
        suite_id="session-lifecycle",
        suite_version="1.0.0",
        decisions=("pass", "review"),
        cases=(Case(case_id="A", input={}), Case(case_id="B", input={})),
        tools=(spec,),
    )
    registry = ToolRegistry()
    effects = []
    sessions = []
    tasks = []
    started = []

    @registry.tool(spec)
    async def increment_counter():
        effects.append("synthetic increment")
        if behavior == "pending":
            started[-1].set()
            await asyncio.Event().wait()
        return {"ok": True}

    manifest = build_manifest(
        suite,
        provider="local",
        model="synthetic",
        adapter="test.session_lifecycle",
        request_parameters={"seed": 42},
        git_sha="f" * 40,
    )

    @agent(manifest=manifest, suite=suite, tools=registry)
    async def candidate(case, context):
        sessions.append(context.tools)
        if behavior == "error":
            raise RuntimeError("synthetic failure")
        if behavior == "wait":
            await asyncio.Event().wait()
        if behavior == "awaited":
            await context.tools.call("increment_counter")
        else:
            started.append(asyncio.Event())
            tasks.append(asyncio.create_task(context.tools.call("increment_counter")))
            if behavior == "pending":
                await started[-1].wait()
        return AgentResult(
            output_text="DECISION: PASS",
            trajectory=context.tools.trajectory(),
            wire_request=WireRequest.from_payload(
                provider="local",
                model="synthetic",
                adapter="test.session_lifecycle",
                parameters={"seed": 42},
                payload={"seed": 42, "case_id": case.case_id},
            ),
        )

    return candidate, suite, effects, sessions, tasks


def assert_closed(sessions):
    async def late_calls():
        for session in sessions:
            with pytest.raises(ToolExecutionError, match="session is closed"):
                await session.call("increment_counter")

    anyio.run(late_calls)


def test_detached_calls_cannot_execute_after_the_recorded_snapshot(tmp_path):
    candidate, suite, effects, sessions, tasks = lifecycle_agent("detached")
    replay = Replay(
        suite=suite,
        replays=2,
        out=tmp_path / "run",
        gate=GatePolicy(min_dar=1, min_tar_seq=1),
        mode="blocking",
    )
    report = replay.run(candidate)
    failures = [task.exception() for task in tasks]
    assert len(failures) == 4
    assert all(isinstance(error, ToolExecutionError) for error in failures)
    assert effects == []
    assert all(not session.calls for session in sessions)
    assert report.artifacts_verified
    assert replay.last_gate_result.passed
    assert_closed(sessions)


def test_awaited_calls_are_still_captured_before_closure(tmp_path):
    candidate, suite, effects, sessions, _tasks = lifecycle_agent("awaited")
    report = Replay(suite=suite, replays=2, out=tmp_path / "run").run(candidate)
    with FileStore(tmp_path / "run", create=False) as store:
        episodes = store.list(manifest_hash=candidate.manifest.hash)
    assert len(effects) == 4
    assert report.status.value == "complete"
    assert all(execution_summary(row.trajectory).completed_invocations == 1 for row in episodes)
    assert_closed(sessions)
    assert len(effects) == 4


def test_already_started_calls_remain_unresolved_at_agent_return(tmp_path):
    candidate, suite, effects, sessions, tasks = lifecycle_agent("pending")
    report = Replay(suite=suite, replays=2, out=tmp_path / "run").run(candidate)
    with FileStore(tmp_path / "run", create=False) as store:
        episodes = store.list(manifest_hash=candidate.manifest.hash)
    assert len(effects) == 4
    assert report.status.value == "partial"
    assert report.observed_groups == 0
    for row in episodes:
        summary = execution_summary(row.trajectory)
        assert row.status.value == "tool_error"
        assert summary.unresolved_proposals == 1
        assert summary.completed_invocations == 0
    assert all(task.cancelled() for task in tasks)
    assert_closed(sessions)


@pytest.mark.parametrize("behavior", ["error", "wait"])
def test_agent_failure_or_timeout_closes_its_session(tmp_path, behavior):
    candidate, suite, effects, sessions, _tasks = lifecycle_agent(behavior)
    report = Replay(suite=suite, replays=2, out=tmp_path / "run", episode_timeout_s=0.02).run(
        candidate
    )
    assert report.status.value == "partial"
    assert report.observed_groups == 0
    assert sessions
    assert_closed(sessions)
    assert effects == []


def test_external_cancellation_closes_the_active_session(tmp_path):
    candidate, suite, effects, sessions, _tasks = lifecycle_agent("wait")

    async def cancel_run():
        with anyio.move_on_after(0.02) as scope:
            await Replay(suite=suite, replays=2, out=tmp_path / "run").arun(candidate)
        assert scope.cancel_called

    anyio.run(cancel_run)
    assert sessions
    assert_closed(sessions)
    assert effects == []


def test_call_waiting_to_reserve_cannot_cross_session_closure():
    registry = ToolRegistry()
    spec = ToolSpec(name="increment", input_schema={"type": "object"})
    effects = []
    registry.register(spec, lambda: effects.append("increment"))
    session = registry.session()

    async def queued_call():
        async with session._lock:
            task = asyncio.create_task(session.call("increment"))
            await anyio.sleep(0)
            session.close()
        with pytest.raises(ToolExecutionError, match="session is closed"):
            await task

    anyio.run(queued_call)
    assert effects == []
    assert session.calls == ()
