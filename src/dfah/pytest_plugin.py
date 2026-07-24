"""Pytest fixtures that bring DFAH gates into an existing test suite."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

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
        help="Optional YAML/JSON DFAH gate policy",
    )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "dfah: replay-stability assertion")


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
