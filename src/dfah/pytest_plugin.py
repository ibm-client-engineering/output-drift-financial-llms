"""Pytest fixtures that bring DFAH gates into an existing test suite."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from yaml import YAMLError  # type: ignore[import-untyped]

from .exceptions import DFAHError
from .gate import Gate, GatePolicy, GateResult
from .models import Report


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("dfah")
    group.addoption(
        "--dfah-report",
        action="store",
        default=None,
        help="Path to a DFAH run directory or report inside its reports directory",
    )
    group.addoption(
        "--dfah-policy",
        action="store",
        default=None,
        help="Enforce a YAML/JSON DFAH gate policy before test collection; requires --dfah-report",
    )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "dfah: replay-stability assertion")


def pytest_sessionstart(session: Any) -> None:
    """Enforce an explicitly requested policy even without DFAH fixture users."""

    config = session.config
    configured = config.getoption("--dfah-policy")
    if configured is None:
        return
    report_path = config.getoption("--dfah-report")
    if not report_path:
        raise pytest.UsageError("--dfah-policy requires --dfah-report")
    try:
        policy = GatePolicy.load(configured)
        report = Report.from_json(Path(report_path))
        result = Gate(policy).evaluate(report)
    except (DFAHError, OSError, ValueError, YAMLError) as exc:
        # Parser/validation exceptions may contain input values. Keep the
        # session diagnostic content-free, as the portable report is.
        raise pytest.UsageError(
            "DFAH policy preflight requires a valid policy and verified report "
            f"({type(exc).__name__})"
        ) from None
    failures = [check.name for check in result.checks if not check.passed]
    if failures:
        pytest.exit(
            "DFAH gate failed: " + ", ".join(failures),
            returncode=pytest.ExitCode.TESTS_FAILED,
        )
    terminal = config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write_line("DFAH policy passed (verified report)")


@pytest.fixture
def dfah_report(request: Any) -> Report:
    """Load the report supplied with ``--dfah-report``."""

    path = request.config.getoption("--dfah-report")
    if not path:
        pytest.skip("pass --dfah-report to enable DFAH assertions")
    return Report.from_json(Path(path))


@pytest.fixture
def dfah_gate(request: Any, dfah_report: Report) -> Callable[[GatePolicy | None], GateResult]:
    """Return a callable that evaluates a policy against ``dfah_report``."""

    configured = request.config.getoption("--dfah-policy")

    def evaluate(policy: GatePolicy | None = None) -> GateResult:
        selected = policy or (GatePolicy.load(configured) if configured else GatePolicy())
        return Gate(selected).evaluate(dfah_report)

    return evaluate
