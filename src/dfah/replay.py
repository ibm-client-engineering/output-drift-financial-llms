"""Async-first replay orchestration with a synchronous golden path."""

from __future__ import annotations

import asyncio
import math
import random
import re
import subprocess
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anyio

from ._canonical import sha256
from ._meta import package_version
from .agents import Agent, AgentResult, RunContext
from .exceptions import (
    AgentContractError,
    ArtifactError,
    ConfigurationError,
    ToolExecutionError,
)
from .gate import Gate, GatePolicy
from .metrics.agreement import (
    case_reports_from_episodes,
    ineligibility_summary_from_episodes,
    task_weighted,
)
from .models import (
    Case,
    ChannelState,
    DispatchState,
    Episode,
    EpisodeError,
    EpisodeStatus,
    Manifest,
    ParseProvenance,
    ReplayMode,
    Report,
    ReportStatus,
    ToolExecutionState,
    Trajectory,
    utc_now,
)
from .parse import parse_decision
from .store import DispatchIntent, EpisodeStart, FileStore, RunPlan
from .suite import Suite
from .telemetry import episode_span, finish_episode_span
from .telemetry import preflight as telemetry_preflight
from .tools import ToolRegistry


def _git_sha() -> str | None:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if dirty:
        raise ConfigurationError(
            "worktree is dirty; pass an explicit reviewed revision or implementation hash"
        )
    return revision


def build_manifest(
    suite: Suite,
    *,
    provider: str,
    model: str,
    adapter: str,
    request_parameters: dict[str, Any],
    endpoint: str | None = None,
    adapter_version: str | None = None,
    library_version: str | None = None,
    git_sha: str | None = None,
    implementation_hash: str | None = None,
    capture_raw_content: bool = False,
) -> Manifest:
    """Build a pinned manifest from the exact post-normalization request settings."""

    source_revision = git_sha
    if source_revision is None:
        try:
            source_revision = _git_sha()
        except ConfigurationError:
            if implementation_hash is None:
                raise
    return Manifest(
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        decision_ontology=tuple(label.strip().lower() for label in suite.decisions),
        provider=provider,
        model=model,
        endpoint=endpoint,
        temperature=request_parameters.get("temperature"),
        top_p=request_parameters.get("top_p"),
        seed=request_parameters.get("seed"),
        tool_schema_hash=suite.tool_schema_hash,
        fixture_hash=suite.fixture_hash,
        request_parameters=request_parameters,
        adapter=adapter,
        adapter_version=adapter_version,
        library_version=library_version or package_version(),
        git_sha=source_revision,
        implementation_hash=implementation_hash,
        capture_raw_content=capture_raw_content,
    )


class _Budget:
    def __init__(
        self,
        cap: float | None,
        reservation: float | None,
        *,
        initial_spent: float = 0.0,
    ):
        if cap is not None and (not math.isfinite(cap) or cap <= 0):
            raise ConfigurationError("budget_usd must be positive")
        if cap is not None and (
            reservation is None or not math.isfinite(reservation) or reservation <= 0
        ):
            raise ConfigurationError(
                "a hard budget requires estimated_max_episode_cost_usd for reservation"
            )
        if reservation is not None and (not math.isfinite(reservation) or reservation < 0):
            raise ConfigurationError("episode cost reservation must be finite and nonnegative")
        self.cap = cap
        self.reservation = reservation or 0.0
        if not math.isfinite(initial_spent) or initial_spent < 0:
            raise ConfigurationError("initial budget spend cannot be negative")
        self.spent = initial_spent
        self.reserved = 0.0
        self.exceeded = cap is not None and initial_spent > cap
        self.lock = anyio.Lock()

    async def reserve(self) -> bool:
        async with self.lock:
            if (
                self.cap is not None
                and self.spent + self.reserved + self.reservation > self.cap
            ):
                return False
            self.reserved += self.reservation
            return True

    async def settle(self, observed: float) -> None:
        async with self.lock:
            self.reserved = max(0.0, self.reserved - self.reservation)
            self.spent += observed
            if self.cap is not None and self.spent > self.cap + 1e-12:
                self.exceeded = True

    async def release(self) -> None:
        async with self.lock:
            self.reserved = max(0.0, self.reserved - self.reservation)


class Replay:
    """Run a versioned suite repeatedly against any conforming agent.

    The default is a serial, no-retry shadow evaluation. Complete episodes are
    reused idempotently. A durable start without a completion is retained as an
    interrupted episode and is never silently dispatched again.
    """

    def __init__(
        self,
        *,
        suite: Suite | str | Path,
        replays: int = 3,
        seed: int = 42,
        out: str | Path | None = None,
        concurrency: int = 1,
        sample_rate: float = 1.0,
        mode: ReplayMode | str = ReplayMode.SHADOW,
        budget_usd: float | None = None,
        estimated_max_episode_cost_usd: float | None = None,
        gate: GatePolicy | None = None,
        otel: bool = False,
        tools: ToolRegistry | None = None,
        capture_tool_arguments: bool = False,
        recover_stale_lease: bool = False,
        episode_timeout_s: float | None = None,
    ):
        if replays < 2:
            raise ConfigurationError("replays must be at least 2")
        if concurrency < 1:
            raise ConfigurationError("concurrency must be at least 1")
        if not 0.0 < sample_rate <= 1.0:
            raise ConfigurationError("sample_rate must be in (0, 1]")
        if episode_timeout_s is not None and (
            not math.isfinite(episode_timeout_s) or episode_timeout_s <= 0
        ):
            raise ConfigurationError("episode_timeout_s must be finite and positive")
        self.suite = Suite.load(suite)
        self.replays = replays
        self.seed = seed
        self.out = Path(out) if out is not None else None
        self.concurrency = concurrency
        self.sample_rate = sample_rate
        self.mode = ReplayMode(mode)
        self.budget_usd = budget_usd
        self.estimated_max_episode_cost_usd = estimated_max_episode_cost_usd
        self.gate_policy = gate
        self.otel = otel
        self.tools = tools
        self.capture_tool_arguments = capture_tool_arguments
        self.recover_stale_lease = recover_stale_lease
        self.episode_timeout_s = episode_timeout_s
        if tools is not None and tools.schema_hash != self.suite.tool_schema_hash:
            raise ConfigurationError("tool registry schemas differ from the selected suite")

    def output_path(self, manifest: Manifest) -> Path:
        """Return the explicit path or a deterministic contract/design-specific default."""

        if self.out is not None:
            return self.out
        suite_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", self.suite.suite_id).strip("-.")
        suite_slug = suite_slug[:48] or "suite"
        design_id = sha256(
            {
                "manifest_hash": manifest.hash,
                "replays": self.replays,
                "seed": self.seed,
                "sample_rate": self.sample_rate,
                "episode_timeout_s": self.episode_timeout_s,
            }
        )[:12]
        return Path(".dfah/runs") / f"{suite_slug}-{design_id}"

    def _validate_manifest(self, manifest: Manifest) -> None:
        expected = (
            self.suite.suite_id,
            self.suite.suite_version,
            tuple(label.strip().lower() for label in self.suite.decisions),
            self.suite.fixture_hash,
            self.suite.tool_schema_hash,
        )
        observed = (
            manifest.suite_id,
            manifest.suite_version,
            manifest.decision_ontology,
            manifest.fixture_hash,
            manifest.tool_schema_hash,
        )
        if observed != expected:
            raise ConfigurationError(
                "agent manifest does not match suite ID/version/ontology/fixtures/tools"
            )

    def _validated_external_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """Validate and redact manual tool capture before it can be persisted."""

        if not trajectory.tool_calls:
            return trajectory
        specs = {spec.name: spec for spec in self.suite.tools}
        sanitized = []
        for call in trajectory.tool_calls:
            spec = specs.get(call.name)
            if spec is None:
                raise AgentContractError(
                    f"captured tool {call.name!r} is not declared by the selected suite"
                )
            if call.arguments_redacted:
                raise AgentContractError(
                    f"captured tool {call.name!r} redacted arguments before schema validation"
                )
            try:
                spec.validate_arguments(dict(call.arguments))
            except ValueError as exc:
                raise AgentContractError(f"captured tool {call.name!r}: {exc}") from None
            sanitized.append(
                call
                if self.capture_tool_arguments
                else call.model_copy(
                    update={
                        "arguments": {},
                        "arguments_hash": call.argument_hash,
                        "arguments_redacted": True,
                    }
                )
            )
        return trajectory.model_copy(update={"tool_calls": tuple(sanitized)})

    def _selected_cases(self) -> tuple[Case, ...]:
        if self.sample_rate == 1.0:
            return tuple(sorted(self.suite.cases, key=lambda case: case.effective_case_id))
        rng = random.Random(self.seed)
        count = max(1, round(len(self.suite.cases) * self.sample_rate))
        selected = rng.sample(list(self.suite.cases), count)
        return tuple(sorted(selected, key=lambda case: case.effective_case_id))

    @staticmethod
    def _episode_id(manifest_hash: str, case_id: str, replay_index: int) -> str:
        return sha256(
            {"manifest_hash": manifest_hash, "case_id": case_id, "replay_index": replay_index}
        )[:32]

    def _interrupted_episode(
        self,
        start: EpisodeStart,
        *,
        reason: str,
        reserved_cost_usd: float,
    ) -> Episode:
        ended = utc_now()
        return Episode(
            episode_id=self._episode_id(*start.key),
            manifest_hash=start.manifest_hash,
            suite_id=start.suite_id,
            suite_version=start.suite_version,
            case_id=start.case_id,
            task=start.task,
            replay_index=start.replay_index,
            status=EpisodeStatus.INTERRUPTED,
            dispatch_state=DispatchState.UNKNOWN_AFTER_DISPATCH,
            decision=None,
            parse=ParseProvenance(strategy="none", fallback=True, accepted=False),
            trajectory=Trajectory(state=ChannelState.UNAVAILABLE),
            cost_usd=reserved_cost_usd,
            started_at=start.started_at,
            ended_at=max(ended, start.started_at),
            error=EpisodeError(
                kind="unknown_after_dispatch",
                message=reason,
                retryable=False,
                dispatch_state=DispatchState.UNKNOWN_AFTER_DISPATCH,
            ),
        )

    def _settle_durable_schedule(
        self,
        *,
        manifest: Manifest,
        schedule: Sequence[tuple[Case, int]],
        store: FileStore,
    ) -> tuple[Episode, ...]:
        """Recover/settle every prior schedule state before admitting new work."""

        expected = {
            (case.effective_case_id, replay_index): case.task for case, replay_index in schedule
        }
        store.assert_schedule_inventory(
            manifest_hash=manifest.hash,
            suite_id=self.suite.suite_id,
            suite_version=self.suite.suite_version,
            expected=expected,
        )
        for case, replay_index in schedule:
            case_id = case.effective_case_id
            state = store.inspect(manifest.hash, case_id, replay_index)
            if state == "corrupt":
                recovered = store.recover_partial(manifest.hash, case_id, replay_index)
                if recovered is None:
                    raise ArtifactError(f"corrupt episode state for {case_id}/{replay_index}")
            elif state == "dispatching":
                start = store.read_start(manifest.hash, case_id, replay_index)
                dispatch = store.read_dispatch(manifest.hash, case_id, replay_index)
                store.commit(
                    self._interrupted_episode(
                        start,
                        reason=(
                            "prior process ended after the durable dispatch boundary; the "
                            "episode was not reissued"
                        ),
                        reserved_cost_usd=dispatch.reserved_cost_usd,
                    )
                )

        committed = store.list(manifest_hash=manifest.hash)
        unexpected = [
            (episode.case_id, episode.replay_index)
            for episode in committed
            if (episode.case_id, episode.replay_index) not in expected
        ]
        if unexpected:
            raise ConfigurationError(
                "run directory contains committed episodes outside the current replay design"
            )
        return committed

    async def _one(
        self,
        *,
        agent: Agent,
        case: Case,
        replay_index: int,
        store: FileStore,
        budget: _Budget,
        tools: ToolRegistry | None,
    ) -> Episode:
        manifest = agent.manifest
        case_id = case.effective_case_id
        state = store.inspect(manifest.hash, case_id, replay_index)
        if state == "committed":
            return store.read(manifest.hash, case_id, replay_index)
        if state == "corrupt":
            recovered = store.recover_partial(manifest.hash, case_id, replay_index)
            if recovered is not None:
                return recovered
            raise ArtifactError(f"corrupt episode state for {case_id}/{replay_index}")
        if state == "dispatching":
            start = store.read_start(manifest.hash, case_id, replay_index)
            dispatch = store.read_dispatch(manifest.hash, case_id, replay_index)
            await budget.settle(dispatch.reserved_cost_usd)
            return store.commit(
                self._interrupted_episode(
                    start,
                    reason=(
                        "prior process ended after the durable dispatch boundary; the "
                        "episode was not reissued"
                    ),
                    reserved_cost_usd=dispatch.reserved_cost_usd,
                )
            )

        if state == "started":
            # A planned-only episode is provably pre-dispatch and safe to resume.
            start = store.read_start(manifest.hash, case_id, replay_index)
            started_at = start.started_at
            idempotency_key = start.idempotency_key
        else:
            started_at = utc_now()
            idempotency_key = sha256(
                {"manifest": manifest.hash, "case": case_id, "replay": replay_index}
            )
            start = EpisodeStart(
                manifest_hash=manifest.hash,
                suite_id=self.suite.suite_id,
                suite_version=self.suite.suite_version,
                case_id=case_id,
                task=case.task,
                replay_index=replay_index,
                started_at=started_at,
                idempotency_key=idempotency_key,
            )
            store.start(start)

        if not await budget.reserve():
            # No external side effect occurred. Keep the durable plan resumable;
            # the partial report records this admission stop for the current run.
            return Episode(
                episode_id=self._episode_id(*start.key),
                manifest_hash=manifest.hash,
                suite_id=self.suite.suite_id,
                suite_version=self.suite.suite_version,
                case_id=case_id,
                task=case.task,
                replay_index=replay_index,
                status=EpisodeStatus.BUDGET_STOP,
                dispatch_state=DispatchState.RESERVED,
                decision=None,
                parse=ParseProvenance(strategy="none", fallback=True, accepted=False),
                trajectory=Trajectory(state=ChannelState.UNAVAILABLE),
                started_at=started_at,
                ended_at=utc_now(),
                error=EpisodeError(
                    kind="budget_stop",
                    message="starting the episode would exceed the declared budget",
                    retryable=False,
                    dispatch_state=DispatchState.RESERVED,
                ),
            )

        tool_session = (
            tools.session(
                id_prefix=self._episode_id(*start.key),
                capture_arguments=self.capture_tool_arguments,
                otel=self.otel,
            )
            if tools is not None
            else None
        )
        context = RunContext(
            manifest_hash=manifest.hash,
            case_id=case_id,
            replay_index=replay_index,
            seed=self.seed,
            idempotency_key=idempotency_key,
            tools=tool_session,
        )
        span_context = episode_span(manifest, enabled=self.otel)
        span = span_context.__enter__()
        try:
            try:
                store.mark_dispatching(
                    DispatchIntent(
                        manifest_hash=manifest.hash,
                        case_id=case_id,
                        replay_index=replay_index,
                        marked_at=utc_now(),
                        reserved_cost_usd=budget.reservation,
                    )
                )
            except Exception:
                await budget.release()
                raise

            before = time.perf_counter()
            try:
                if self.episode_timeout_s is None:
                    result = await agent.arun(case, context)
                else:
                    with anyio.fail_after(self.episode_timeout_s):
                        result = await agent.arun(case, context)
                if not isinstance(result, AgentResult):
                    raise AgentContractError(
                        "Agent.arun must return dfah.AgentResult; implicit coercion is disabled"
                    )
            except Exception as exc:
                await budget.settle(budget.reservation)
                timed_out = isinstance(exc, TimeoutError)
                message = (
                    "registered tool request failed; inspect the hashed tool-call record"
                    if isinstance(exc, ToolExecutionError)
                    else "agent returned a value that violates the DFAH Agent contract"
                    if isinstance(exc, AgentContractError)
                    else (
                        "episode exceeded its configured timeout after the durable dispatch "
                        "boundary; provider-side completion and cost are unknown"
                    )
                    if timed_out
                    else (
                        "adapter failed after the dispatch boundary; consult provider-side "
                        "diagnostics outside DFAH"
                    )
                )
                episode = Episode(
                    episode_id=self._episode_id(*start.key),
                    manifest_hash=manifest.hash,
                    suite_id=self.suite.suite_id,
                    suite_version=self.suite.suite_version,
                    case_id=case_id,
                    task=case.task,
                    replay_index=replay_index,
                    status=(
                        EpisodeStatus.TOOL_ERROR
                        if isinstance(exc, ToolExecutionError)
                        else EpisodeStatus.CONTRACT_ERROR
                        if isinstance(exc, AgentContractError)
                        else EpisodeStatus.PROVIDER_ERROR
                    ),
                    dispatch_state=DispatchState.UNKNOWN_AFTER_DISPATCH,
                    decision=None,
                    parse=ParseProvenance(strategy="none", fallback=True, accepted=False),
                    trajectory=(
                        tool_session.trajectory()
                        if tool_session is not None
                        else Trajectory(state=ChannelState.UNAVAILABLE)
                    ),
                    cost_usd=budget.reservation,
                    latency_ms=(time.perf_counter() - before) * 1000.0,
                    started_at=started_at,
                    ended_at=utc_now(),
                    error=EpisodeError(
                        kind="episode_timeout" if timed_out else type(exc).__name__,
                        message=message[:2000],
                        retryable=False,
                        dispatch_state=DispatchState.UNKNOWN_AFTER_DISPATCH,
                    ),
                )
                finish_episode_span(span, episode)
                return store.commit(episode)

            try:
                manifest.validate_wire_request(result.wire_request)
                if tool_session is not None and result.trajectory != tool_session.trajectory():
                    raise AgentContractError(
                        "AgentResult trajectory differs from the injected ToolSession record"
                    )
                trajectory = (
                    result.trajectory
                    if tool_session is not None
                    else self._validated_external_trajectory(result.trajectory)
                )
                decision, provenance = parse_decision(result.output_text, self.suite.decisions)
                tool_failure = any(
                    call.execution_state is not ToolExecutionState.EXECUTED
                    for call in trajectory.tool_calls
                )
                if tool_failure:
                    status = EpisodeStatus.TOOL_ERROR
                    error = EpisodeError(
                        kind="tool_capture_incomplete",
                        message=(
                            "one or more requested tools were rejected, failed, or incomplete"
                        ),
                        retryable=False,
                        dispatch_state=DispatchState.RESPONDED,
                    )
                elif decision is None:
                    status = EpisodeStatus.PARSE_FAILURE
                    error = EpisodeError(
                        kind="parse_failure",
                        message=(
                            "no single allowed DECISION marker was emitted; no label "
                            "substituted"
                        ),
                        retryable=False,
                        dispatch_state=DispatchState.RESPONDED,
                    )
                else:
                    status = EpisodeStatus.VALID
                    error = None
                episode = Episode(
                    episode_id=self._episode_id(*start.key),
                    manifest_hash=manifest.hash,
                    suite_id=self.suite.suite_id,
                    suite_version=self.suite.suite_version,
                    case_id=case_id,
                    task=case.task,
                    replay_index=replay_index,
                    status=status,
                    dispatch_state=DispatchState.COMPLETE,
                    decision=decision,
                    parse=provenance,
                    trajectory=trajectory,
                    wire_request=result.wire_request,
                    usage=result.usage,
                    cost_usd=result.cost_usd,
                    latency_ms=(time.perf_counter() - before) * 1000.0,
                    started_at=started_at,
                    ended_at=utc_now(),
                    error=error,
                )
            except Exception as exc:
                message = (
                    "agent response violated the manifest, tool, or capture contract; "
                    "inspect local diagnostics outside the portable artifact"
                )
                episode = Episode(
                    episode_id=self._episode_id(*start.key),
                    manifest_hash=manifest.hash,
                    suite_id=self.suite.suite_id,
                    suite_version=self.suite.suite_version,
                    case_id=case_id,
                    task=case.task,
                    replay_index=replay_index,
                    status=EpisodeStatus.CONTRACT_ERROR,
                    dispatch_state=DispatchState.RESPONDED,
                    decision=None,
                    parse=ParseProvenance(strategy="none", fallback=True, accepted=False),
                    trajectory=Trajectory(state=ChannelState.UNAVAILABLE),
                    wire_request=None,
                    cost_usd=result.cost_usd,
                    latency_ms=(time.perf_counter() - before) * 1000.0,
                    started_at=started_at,
                    ended_at=utc_now(),
                    error=EpisodeError(
                        kind=type(exc).__name__,
                        message=message[:2000],
                        retryable=False,
                        dispatch_state=DispatchState.RESPONDED,
                    ),
                )
            await budget.settle(result.cost_usd)
            finish_episode_span(span, episode)
            return store.commit(episode)
        finally:
            span_context.__exit__(None, None, None)

    def _report(
        self,
        manifest: Manifest,
        episodes: Sequence[Episode],
        *,
        budget_exceeded: bool,
        budget_stopped: bool,
        episode_artifact_root_sha256: str,
        episode_artifact_count: int,
        run_plan_sha256: str,
    ) -> Report:
        case_reports = case_reports_from_episodes(episodes, required_replays=self.replays)
        eligible_episode_count = sum(row.replay_count for row in case_reports)

        selected_cases = self._selected_cases()
        planned = len(selected_cases) * self.replays
        ineligible_groups, ineligibility_reasons = ineligibility_summary_from_episodes(
            episodes,
            required_replays=self.replays,
            expected_groups=tuple(
                (case.task, case.effective_case_id) for case in selected_cases
            ),
        )
        if case_reports:
            summary = task_weighted(case_reports)
            summary_dar, summary_tar, summary_gap = summary.dar, summary.tar, summary.gap
        else:
            summary_dar = None
            summary_tar = None
            summary_gap = None
        status = (
            ReportStatus.COMPLETE
            if (
                len(episodes) == planned
                and eligible_episode_count == planned
                and not budget_exceeded
            )
            else ReportStatus.PARTIAL
            if episodes or budget_stopped
            else ReportStatus.INELIGIBLE
        )
        partial_reason = None
        if status is not ReportStatus.COMPLETE:
            partial_reason = (
                "budget ceiling exceeded after provider charge; increase the reservation bound"
                if budget_exceeded
                else "budget ceiling reached"
                if budget_stopped
                else "one or more replay groups are ineligible"
            )
        return Report(
            report_id=str(uuid.uuid4()),
            suite_id=self.suite.suite_id,
            suite_version=self.suite.suite_version,
            manifest=manifest,
            status=status,
            replays_requested=self.replays,
            schedule_seed=self.seed,
            sample_rate=self.sample_rate,
            suite_cases_total=len(self.suite.cases),
            cases_selected=planned // self.replays,
            mode=self.mode,
            episodes_planned=planned,
            episodes_completed=len(episodes),
            episodes_eligible=eligible_episode_count,
            case_reports=tuple(case_reports),
            dar=summary_dar,
            tar=summary_tar,
            gap=summary_gap,
            total_cost_usd=sum(episode.cost_usd for episode in episodes),
            budget_exceeded=budget_exceeded,
            total_latency_ms=sum(episode.latency_ms for episode in episodes),
            flagged_groups=sum(row.unanimous_with_path_change for row in case_reports),
            observed_groups=len(case_reports),
            ineligible_groups=ineligible_groups,
            ineligibility_reasons=ineligibility_reasons,
            partial_reason=partial_reason,
            run_dir=None,
            episode_artifact_root_sha256=episode_artifact_root_sha256,
            episode_artifact_count=episode_artifact_count,
            run_plan_sha256=run_plan_sha256,
        )

    async def _arun_locked(
        self,
        agent: Agent,
        store: FileStore,
        tools: ToolRegistry | None,
        out: Path,
    ) -> Report:
        """Execute while the caller holds the artifact directory writer lease."""

        selected = self._selected_cases()
        schedule = [(case, replay) for case in selected for replay in range(self.replays)]
        random.Random(self.seed).shuffle(schedule)
        run_plan = RunPlan(
            manifest_hash=agent.manifest.hash,
            suite_id=self.suite.suite_id,
            suite_version=self.suite.suite_version,
            fixture_hash=self.suite.fixture_hash,
            tool_schema_hash=self.suite.tool_schema_hash,
            replays=self.replays,
            seed=self.seed,
            sample_rate=self.sample_rate,
            episode_timeout_s=self.episode_timeout_s,
            case_tasks={case.effective_case_id: case.task for case in selected},
            schedule_sha256=sha256(
                tuple(
                    {
                        "case_id": case.effective_case_id,
                        "task": case.task,
                        "replay_index": replay_index,
                    }
                    for case, replay_index in schedule
                )
            ),
            suite_cases_total=len(self.suite.cases),
            episodes_planned=len(schedule),
        )
        store.bind_plan(run_plan)
        prior = self._settle_durable_schedule(
            manifest=agent.manifest,
            schedule=schedule,
            store=store,
        )
        budget = _Budget(
            self.budget_usd,
            self.estimated_max_episode_cost_usd,
            initial_spent=sum(episode.cost_usd for episode in prior),
        )
        results: list[Episode] = []
        result_lock = anyio.Lock()

        send_stream, receive_stream = anyio.create_memory_object_stream[tuple[Case, int]](
            max_buffer_size=max(1, self.concurrency * 2)
        )

        async def produce() -> None:
            async with send_stream:
                for item in schedule:
                    await send_stream.send(item)

        async def worker() -> None:
            async with receive_stream.clone() as queue:
                async for case, replay_index in queue:
                    episode = await self._one(
                        agent=agent,
                        case=case,
                        replay_index=replay_index,
                        store=store,
                        budget=budget,
                        tools=tools,
                    )
                    async with result_lock:
                        results.append(episode)

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(produce)
            for _ in range(self.concurrency):
                tasks.start_soon(worker)
        await receive_stream.aclose()

        committed = store.list(manifest_hash=agent.manifest.hash)
        artifact_root, artifact_count = store.commitment(manifest_hash=agent.manifest.hash)
        report = self._report(
            agent.manifest,
            committed,
            budget_exceeded=budget.exceeded,
            budget_stopped=any(
                episode.status is EpisodeStatus.BUDGET_STOP for episode in results
            ),
            episode_artifact_root_sha256=artifact_root,
            episode_artifact_count=artifact_count,
            run_plan_sha256=run_plan.hash,
        )
        report.verify_artifacts(out)
        reports_dir = out / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        report.to_json(reports_dir / f"{report.report_id}.json")
        if self.gate_policy is not None:
            gate_result = Gate(self.gate_policy).evaluate(report)
            if self.mode is ReplayMode.BLOCKING:
                gate_result.raise_for_failures()
        return report

    async def arun(self, agent: Agent) -> Report:
        """Execute or resume a replay asynchronously."""

        if not isinstance(agent, Agent):
            raise AgentContractError("agent does not implement the dfah.Agent protocol")
        self._validate_manifest(agent.manifest)
        embedded_tools = getattr(agent, "tools", None)
        if embedded_tools is not None and not isinstance(embedded_tools, ToolRegistry):
            raise AgentContractError("agent.tools must be a dfah.ToolRegistry")
        if (
            self.tools is not None
            and embedded_tools is not None
            and self.tools is not embedded_tools
        ):
            raise ConfigurationError("Replay tools override conflicts with agent-bound tools")
        tools = self.tools or embedded_tools
        if tools is not None and tools.schema_hash != self.suite.tool_schema_hash:
            raise ConfigurationError("tool registry schemas differ from the selected suite")
        if tools is not None:
            tools.freeze()
        telemetry_preflight(enabled=self.otel)
        out = self.output_path(agent.manifest)
        with (
            FileStore(out) as store,
            store.run_lease(
                manifest_hash=agent.manifest.hash,
                recover_stale=self.recover_stale_lease,
            ),
        ):
            return await self._arun_locked(agent, store, tools, out)

    def run(self, agent: Agent) -> Report:
        """Synchronous facade for scripts, notebooks, and CI."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return anyio.run(self.arun, agent)
        raise RuntimeError(
            "Replay.run() cannot run inside an event loop; use `await Replay.arun()`"
        )
