"""Local package checks matching the DFAH CI matrix."""

from __future__ import annotations

import nox


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests(session: nox.Session) -> None:
    """Run typed package tests without any live provider calls."""

    session.install("-e", ".[dev,otel]")
    session.run("pytest", "-q", "tests/dfah")
    session.run("python", "examples/dfah_execution_evidence.py")


@nox.session
def quality(session: nox.Session) -> None:
    """Run formatting, lint, and strict type checks."""

    session.install("-e", ".[dev]")
    paths = (
        "src/dfah",
        "tests/dfah",
        "examples/dfah_quickstart.py",
        "examples/dfah_gate_loop.py",
        "examples/dfah_execution_evidence.py",
        "hatch_build.py",
    )
    session.run("ruff", "format", "--check", *paths)
    session.run("ruff", "check", *paths)
    session.run("mypy", "src/dfah")
