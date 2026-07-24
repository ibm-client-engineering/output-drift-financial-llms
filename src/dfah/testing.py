"""One-line integration conformance checks for user agents."""

from __future__ import annotations

import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from ._canonical import redact_text
from .agents import Agent, AgentResult, RunContext
from .models import (
    Case,
    ConformanceCheck,
    ConformanceReport,
    ConformanceStatus,
    Episode,
    Manifest,
)
from .replay import Replay
from .store import FileStore
from .suite import Suite


class _CountingAgent:
    def __init__(self, inner: Agent) -> None:
        self.inner = inner
        self.calls = 0

    @property
    def manifest(self) -> Manifest:
        return self.inner.manifest

    async def arun(self, case: Case, context: RunContext) -> AgentResult:
        self.calls += 1
        return await self.inner.arun(case, context)


@contextmanager
def _diagnostic_directory(out: str | Path | None) -> Iterator[tuple[Path, bool]]:
    if out is not None:
        yield Path(out).expanduser().resolve(), True
        return
    with tempfile.TemporaryDirectory(prefix="dfah-conformance-") as directory:
        yield Path(directory).resolve(), False


def _terminal_summary(episodes: Sequence[Episode]) -> str:
    statuses = Counter(episode.status.value for episode in episodes)
    errors = Counter(episode.error.kind for episode in episodes if episode.error is not None)
    status_text = ", ".join(f"{name}={count}" for name, count in sorted(statuses.items()))
    error_text = ", ".join(f"{name}={count}" for name, count in sorted(errors.items()))
    summary = f"terminal statuses: {status_text or 'none'}"
    return f"{summary}; error kinds: {error_text}" if error_text else summary


def check_agent(
    candidate: Agent,
    *,
    suite: Suite | str | Path | None = None,
    max_cases: int = 2,
    budget_usd: float | None = None,
    estimated_max_episode_cost_usd: float | None = None,
    episode_timeout_s: float | None = None,
    out: str | Path | None = None,
    raise_on_error: bool = False,
) -> ConformanceReport:
    """Run a tiny integration suite before a larger paid replay.

    The supplied suite must match the agent manifest. If omitted, DFAH loads a
    built-in suite with the manifest's suite ID. The check performs two replays
    per case, then runs the same replay again to prove committed episodes are
    reused without invoking the agent. Contract and integration-readiness
    violations fail conformance. Observed decision or path variation is
    diagnostic: it is reported as a warning without treating expected model
    nondeterminism as an integration failure. This bounded smoke test does not
    prove the absence of wall-clock or other ambient-state dependencies.
    """

    if max_cases < 1:
        raise ValueError("max_cases must be at least 1")
    checks: list[ConformanceCheck] = []
    try:
        selected_suite = Suite.load(
            suite or getattr(candidate, "suite", None) or candidate.manifest.suite_id
        )
        checks.append(
            ConformanceCheck(
                name="suite_load", status=ConformanceStatus.PASS, detail="suite loaded"
            )
        )
    except Exception as exc:
        result = ConformanceReport(
            checks=(
                ConformanceCheck(
                    name="suite_load", status=ConformanceStatus.FAIL, detail=str(exc)
                ),
            )
        )
        if raise_on_error:
            result.raise_for_failures()
        return result

    selected_count = min(max_cases, len(selected_suite.cases))
    planned_episodes = selected_count * 2
    estimated_ceiling = (
        planned_episodes * estimated_max_episode_cost_usd
        if estimated_max_episode_cost_usd is not None
        else None
    )
    wrapped = _CountingAgent(candidate)
    with _diagnostic_directory(out) as (directory, retained):
        runner = Replay(
            suite=selected_suite,
            replays=2,
            seed=19,
            out=directory,
            concurrency=1,
            sample_rate=selected_count / len(selected_suite.cases),
            budget_usd=budget_usd,
            estimated_max_episode_cost_usd=estimated_max_episode_cost_usd,
            episode_timeout_s=episode_timeout_s,
            tools=getattr(candidate, "tools", None),
        )
        try:
            first = runner.run(wrapped)
            first_calls = wrapped.calls
            second = runner.run(wrapped)
            checks.append(
                ConformanceCheck(
                    name="resumability_idempotence",
                    status=(
                        ConformanceStatus.PASS
                        if wrapped.calls == first_calls
                        else ConformanceStatus.FAIL
                    ),
                    detail=(
                        "second run reused committed episode keys"
                        if wrapped.calls == first_calls
                        else "second run invoked the agent again"
                    ),
                )
            )
            with FileStore(directory) as diagnostics_store:
                episodes = diagnostics_store.list(manifest_hash=candidate.manifest.hash)
            terminal_summary = _terminal_summary(episodes)
            retained_note = (
                f"; diagnostics retained at {directory}"
                if retained and directory.exists()
                else ""
            )
            checks.append(
                ConformanceCheck(
                    name="parse_provenance",
                    status=(
                        ConformanceStatus.PASS
                        if episodes and all(episode.parse is not None for episode in episodes)
                        else ConformanceStatus.FAIL
                    ),
                    detail="every episode contains parse provenance",
                )
            )
            echoes = [episode for episode in episodes if episode.wire_request is not None]
            wire_ok = bool(echoes)
            for episode in echoes:
                request = episode.wire_request
                if request is None:
                    wire_ok = False
                    continue
                try:
                    candidate.manifest.validate_wire_request(request)
                except ValueError:
                    wire_ok = False
            checks.append(
                ConformanceCheck(
                    name="wire_manifest_echo",
                    status=(ConformanceStatus.PASS if wire_ok else ConformanceStatus.FAIL),
                    detail=(
                        "manifest matches post-normalization request echoes"
                        if wire_ok
                        else "wire request echo is absent or differs from the manifest"
                    ),
                )
            )
            hashes: dict[tuple[str, str], set[str | None]] = defaultdict(set)
            for episode in episodes:
                for call in episode.trajectory.tool_calls:
                    hashes[(call.name, call.argument_hash)].add(call.output_hash)
            deterministic = not hashes or all(
                len(values) <= 1 and None not in values for values in hashes.values()
            )
            checks.append(
                ConformanceCheck(
                    name="deterministic_tool_outputs",
                    status=(
                        ConformanceStatus.PASS
                        if deterministic and hashes
                        else ConformanceStatus.SKIP
                        if deterministic
                        else ConformanceStatus.FAIL
                    ),
                    detail=(
                        "identical tool inputs produced identical result hashes"
                        if deterministic and hashes
                        else "suite made no tool calls; check is not applicable"
                        if deterministic
                        else ("identical tool inputs changed output or lacked a result hash")
                    ),
                )
            )
            stable_replays = bool(first.case_reports) and all(
                row.dar == 1.0 and row.tar.strong == 1.0 for row in first.case_reports
            )
            checks.append(
                ConformanceCheck(
                    name="replay_stability_smoke",
                    status=(
                        ConformanceStatus.PASS if first.case_reports else ConformanceStatus.SKIP
                    ),
                    detail=(
                        "no replay-visible decision or strong-path variation from ambient state"
                        if stable_replays
                        else f"not evaluated because no replay group was eligible; {terminal_summary}"
                        if not first.case_reports
                        else (
                            "WARNING: decision or strong-path variation was observed; this is "
                            "an observational model result, not an integration-readiness "
                            "failure. Investigate wall-clock and other ambient-state inputs "
                            "before attributing the variation to the model"
                        )
                    ),
                )
            )
            checks.append(
                ConformanceCheck(
                    name="eligible_capture",
                    status=(
                        ConformanceStatus.PASS
                        if first.status.value == "complete"
                        and second.status.value == "complete"
                        and first.episodes_eligible == planned_episodes
                        and second.episodes_eligible == planned_episodes
                        else ConformanceStatus.FAIL
                    ),
                    detail=(
                        "all planned episodes belong to eligible replay groups"
                        if first.status.value == "complete"
                        and second.status.value == "complete"
                        and first.episodes_eligible == planned_episodes
                        and second.episodes_eligible == planned_episodes
                        else (
                            f"capture is incomplete ({first.episodes_eligible}/"
                            f"{planned_episodes} eligible); {terminal_summary}{retained_note}"
                        )
                    ),
                )
            )
        except Exception as exc:
            detail = redact_text(str(exc))[:1000]
            retained_note = (
                f"; diagnostics retained at {directory}"
                if retained and directory.exists()
                else ""
            )
            checks.append(
                ConformanceCheck(
                    name="execution",
                    status=ConformanceStatus.FAIL,
                    detail=(
                        f"{type(exc).__name__}: {detail or 'conformance execution failed'}"
                        f"{retained_note}"
                    ),
                )
            )
    report = ConformanceReport(
        checks=tuple(checks),
        cases_selected=selected_count,
        episodes_planned=planned_episodes,
        estimated_cost_ceiling_usd=estimated_ceiling,
    )
    if raise_on_error:
        report.raise_for_failures()
    return report
