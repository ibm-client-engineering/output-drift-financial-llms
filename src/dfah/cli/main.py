"""Typer CLI mirroring the stable Python workflows."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from .. import __version__
from .._canonical import redact_text
from ..exceptions import ConfigurationError, DFAHError
from ..gate import Gate, GatePolicy
from ..models import ReplayMode, Report
from ..replay import Replay
from ..report import format_ineligibility_reasons, format_metric
from ..suite import Suite, list_suites
from ..testing import check_agent

app = typer.Typer(
    no_args_is_help=True,
    invoke_without_command=True,
    help="Replay stability for tool-using AI agents.",
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)
suites_app = typer.Typer(no_args_is_help=True, help="List and validate versioned suites.")
manifest_app = typer.Typer(no_args_is_help=True, help="Inspect pinned replay manifests.")
app.add_typer(suites_app, name="suites")
app.add_typer(manifest_app, name="manifest")
console = Console()
error_console = Console(stderr=True)


def _load_object(reference: str) -> Any:
    try:
        module_name, object_name = reference.split(":", 1)
        return getattr(importlib.import_module(module_name), object_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise typer.BadParameter(
            "expected an importable reference like my.module:agent"
        ) from exc


def _report_path(source: Path) -> Path:
    """Resolve a report JSON directly or select the newest report in a run."""

    if source.is_file():
        return source
    candidates = list((source / "reports").glob("*.json"))
    if not candidates and source.is_dir():
        candidates = list(source.glob("*.json"))
    if not candidates:
        raise typer.BadParameter(f"no DFAH report JSON found under {source}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _metric_summary(report: Report) -> str:
    tar_seq = report.tar.seq if report.tar is not None else None
    return (
        f"DAR={format_metric(report.dar)} "
        f"TARseq={format_metric(tar_seq)} "
        f"gap={format_metric(report.gap)} "
        f"flags/100={format_metric(report.flags_per_100_cases, digits=1)}"
    )


def _print_ineligibility_diagnostics(report: Report) -> None:
    if not report.ineligible_groups:
        return
    console.print(
        f"eligible_groups={report.observed_groups}/{report.cases_selected} "
        f"eligible_episodes={report.episodes_eligible}/{report.episodes_planned} "
        f"reasons={format_ineligibility_reasons(report)}"
    )


@app.callback()
def root(
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()


@suites_app.command("list")
def suites_list() -> None:
    table = Table("Suite", "Version", "Cases", "Purpose")
    for suite in list_suites():
        table.add_row(
            suite.suite_id, suite.suite_version, str(len(suite.cases)), suite.description
        )
    console.print(table)


@suites_app.command("validate")
def suites_validate(source: str) -> None:
    suite = Suite.load(source)
    console.print(
        f"[green]valid[/green] {suite.suite_id}@{suite.suite_version} "
        f"fixture={suite.fixture_hash[:12]} tools={suite.tool_schema_hash[:12]}"
    )


@app.command()
def run(
    agent: str = typer.Option(..., help="Import reference, e.g. my.module:agent"),
    suite: str | None = typer.Option(
        None,
        help="Built-in name or YAML/JSON path; defaults to the agent-bound suite.",
    ),
    replays: int = typer.Option(3, min=2),
    seed: int = 42,
    out: Annotated[
        Path | None,
        typer.Option(
            help="Run directory; defaults to a contract/design-specific .dfah/runs path."
        ),
    ] = None,
    concurrency: int = typer.Option(1, min=1),
    sample_rate: float = typer.Option(1.0, min=0.000001, max=1.0),
    mode: ReplayMode = ReplayMode.SHADOW,
    budget_usd: float | None = typer.Option(None, min=0.000001),
    max_episode_cost_usd: float | None = typer.Option(None, min=0.000001),
    episode_timeout_s: float | None = typer.Option(
        None,
        min=0.000001,
        help="Per-episode ceiling; a dispatched timeout is recorded and never retried.",
    ),
    otel: bool = False,
    recover_stale_lease: bool = typer.Option(
        False,
        help="Recover a dead local writer lease after confirming no run is active.",
    ),
) -> None:
    candidate = _load_object(agent)
    suite_source = suite or getattr(candidate, "suite", None)
    if suite_source is None:
        raise ConfigurationError(
            "no suite was supplied and the imported agent has no bound suite; pass --suite"
        )
    replay = Replay(
        suite=suite_source,
        replays=replays,
        seed=seed,
        out=out,
        concurrency=concurrency,
        sample_rate=sample_rate,
        mode=mode,
        budget_usd=budget_usd,
        estimated_max_episode_cost_usd=max_episode_cost_usd,
        episode_timeout_s=episode_timeout_s,
        otel=otel,
        recover_stale_lease=recover_stale_lease,
    )
    run_path = replay.output_path(candidate.manifest)
    report = replay.run(candidate)
    console.print(f"run={run_path}")
    console.print(f"{_metric_summary(report)} status={report.status.value}")
    _print_ineligibility_diagnostics(report)


@app.command()
def analyze(
    run_path: Path,
    report: Annotated[
        Path | None, typer.Option("--report", help="Write an HTML report.")
    ] = None,
) -> None:
    value = Report.from_json(_report_path(run_path), allow_unverified=True)
    if report is not None:
        value.to_html(report)
    console.print(
        f"{value.suite_id}@{value.suite_version}: {_metric_summary(value)}, "
        f"artifacts={'verified' if value.artifacts_verified else 'unverified'}"
    )
    _print_ineligibility_diagnostics(value)


@app.command("inspect")
def inspect_case(
    run_path: Path,
    case: str = typer.Option(..., "--case", help="Case ID to explain."),
) -> None:
    """Show verified replay-level path differences without raw content."""

    report_path = _report_path(run_path)
    report = Report.from_json(report_path)
    run_dir = run_path if run_path.is_dir() else report_path.parent.parent
    explanation = report.explain_case(run_dir, case)
    table = Table("Replay", "Status", "Decision", "Tool sequence", "Strong identities")
    for episode in explanation.episodes:
        identities = ", ".join(
            f"{call.name}:{call.arguments_hash[:8]}:{(call.output_hash or 'none')[:8]}"
            for call in episode.tool_calls
        )
        table.add_row(
            str(episode.replay_index),
            episode.status.value,
            episode.decision or "—",
            " → ".join(episode.tool_sequence) or "(empty)",
            identities or "(empty)",
        )
    console.print(
        f"{explanation.suite_id}@{explanation.suite_version} · "
        f"{explanation.task}/{explanation.case_id}"
    )
    console.print(table)


@app.command()
def gate(run_path: Path, policy: Path) -> None:
    result = Gate(GatePolicy.load(policy)).evaluate(Report.from_json(_report_path(run_path)))
    for check in result.checks:
        icon = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
        console.print(f"{icon} {check.name}: {check.observed} vs {check.expected}")
    raise typer.Exit(code=0 if result.passed else 1)


@manifest_app.command("show")
def manifest_show(run_path: Path) -> None:
    """Print the embedded manifest and its canonical hash."""

    report = Report.from_json(_report_path(run_path), allow_unverified=True)
    if not report.artifacts_verified:
        console.print(
            "[yellow]warning: episode artifacts were not available for verification[/yellow]"
        )
    console.print(f"manifest_hash={report.manifest.hash}")
    console.print(JSON.from_data(report.manifest.model_dump(mode="json")))


@app.command("check-agent")
def check_agent_command(
    agent: str = typer.Option(..., help="Import reference, e.g. my.module:agent"),
    suite: str | None = typer.Option(None),
    max_cases: int = typer.Option(2, min=1, help="Bound the smoke-test case count."),
    budget_usd: float | None = typer.Option(None, min=0.000001),
    max_episode_cost_usd: float | None = typer.Option(None, min=0.000001),
    episode_timeout_s: float | None = typer.Option(
        None,
        min=0.000001,
        help="Per-episode ceiling; a dispatched timeout is recorded and never retried.",
    ),
    out: Annotated[
        Path | None,
        typer.Option(help="Retain conformance diagnostics at this path; temporary by default."),
    ] = None,
) -> None:
    result = check_agent(
        _load_object(agent),
        suite=suite,
        max_cases=max_cases,
        budget_usd=budget_usd,
        estimated_max_episode_cost_usd=max_episode_cost_usd,
        episode_timeout_s=episode_timeout_s,
        out=out,
    )
    console.print(
        f"planned calls: at most {result.episodes_planned} across "
        f"{result.cases_selected} case(s)"
    )
    for check in result.checks:
        icon = (
            "[yellow]SKIP[/yellow]"
            if check.skipped
            else "[green]PASS[/green]"
            if check.passed
            else "[red]FAIL[/red]"
        )
        console.print(f"{icon} {check.name}: {check.detail}")
    raise typer.Exit(code=0 if result.passed else 1)


@app.command()
def doctor() -> None:
    table = Table("Check", "Status")
    table.add_row("Python", sys.version.split()[0])
    table.add_row("DFAH", __version__)
    table.add_row(
        "FileStore",
        "supported (POSIX)" if os.name == "posix" else "unsupported on this platform",
    )
    for package in ("pydantic", "jsonschema", "opentelemetry"):
        try:
            importlib.import_module(package)
            status = "installed"
        except ImportError:
            status = "optional / not installed"
        table.add_row(package, status)
    console.print(table)


def _expected_error_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        details = []
        for error in exc.errors(include_input=False, include_url=False):
            location = ".".join(str(part) for part in error.get("loc", ())) or "input"
            details.append(f"{location}: {error.get('msg', 'invalid value')}")
        return redact_text("; ".join(details[:5]))
    return redact_text(str(exc))


def main() -> None:
    """Run the CLI without exposing tracebacks or locals for expected user errors."""

    try:
        app()
    except (DFAHError, ValidationError, OSError) as exc:
        error_console.print(f"[red]Error:[/red] {_expected_error_message(exc)}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
