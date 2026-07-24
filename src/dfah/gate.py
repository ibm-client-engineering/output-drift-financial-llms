"""Declarative report gates for CI and shadow-to-blocking promotion."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from .models import Record, Report


class TaskGatePolicy(Record):
    """Optional thresholds applied within one named suite task."""

    task: str = Field(min_length=1)
    min_dar: float | None = Field(default=None, ge=0.0, le=1.0)
    min_tar_seq: float | None = Field(default=None, ge=0.0, le=1.0)
    max_gap: float | None = Field(default=None, ge=-1.0, le=1.0)
    max_unanimous_path_change_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    min_observed_groups: int = Field(default=1, ge=1)


class GatePolicy(Record):
    """Thresholds are optional; only declared checks are evaluated."""

    min_dar: float | None = Field(default=None, ge=0.0, le=1.0)
    min_tar_seq: float | None = Field(default=None, ge=0.0, le=1.0)
    max_gap: float | None = Field(default=None, ge=-1.0, le=1.0)
    max_unanimous_path_change_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_flags_per_100_cases: float | None = Field(default=None, ge=0.0, le=100.0)
    max_cost_per_case_usd: float | None = Field(default=None, ge=0.0)
    required_replays: int | None = Field(default=None, ge=2)
    min_eligible_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    min_observed_groups: int | None = Field(default=None, ge=1)
    require_complete: bool = True
    require_artifact_verification: bool = True
    by_task: tuple[TaskGatePolicy, ...] = ()

    @model_validator(mode="after")
    def _unique_task_policies(self) -> GatePolicy:
        names = [policy.task for policy in self.by_task]
        if len(names) != len(set(names)):
            raise ValueError("by_task contains duplicate task policies")
        return self

    @classmethod
    def load(cls, path: str | Path) -> GatePolicy:
        """Load a JSON or YAML gate policy."""

        source = Path(path)
        if source.suffix.lower() == ".json":
            return cls.model_validate_json(source.read_bytes())
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("YAML policies require PyYAML") from exc
        return cls.model_validate(yaml.safe_load(source.read_text(encoding="utf-8")))


class GateCheck(Record):
    """One threshold comparison."""

    name: str
    passed: bool
    observed: float | str
    expected: float | str


class GateResult(Record):
    """Typed output suitable for a CLI exit code or pytest assertion."""

    passed: bool
    checks: tuple[GateCheck, ...]

    def raise_for_failures(self) -> None:
        """Raise a readable gate violation."""

        from .exceptions import GateViolationError

        failures = [check.name for check in self.checks if not check.passed]
        if failures:
            raise GateViolationError("DFAH gate failed: " + ", ".join(failures))


class Gate:
    """Evaluate one immutable policy against one report."""

    def __init__(self, policy: GatePolicy):
        self.policy = policy

    def evaluate(self, report: Report) -> GateResult:
        """Return all checks; do not short-circuit after the first breach."""

        checks: list[GateCheck] = []

        if self.policy.require_artifact_verification:
            checks.append(
                GateCheck(
                    name="artifact_verification",
                    passed=report.artifacts_verified,
                    observed="verified" if report.artifacts_verified else "unverified",
                    expected="verified",
                )
            )

        def minimum(name: str, observed: float, expected: float | None) -> None:
            if expected is not None:
                checks.append(
                    GateCheck(
                        name=name,
                        passed=observed >= expected,
                        observed=observed,
                        expected=expected,
                    )
                )

        def maximum(name: str, observed: float, expected: float | None) -> None:
            if expected is not None:
                checks.append(
                    GateCheck(
                        name=name,
                        passed=observed <= expected,
                        observed=observed,
                        expected=expected,
                    )
                )

        if report.observed_groups:
            assert report.dar is not None and report.tar is not None and report.gap is not None
            minimum("dar", report.dar, self.policy.min_dar)
            minimum("tar_seq", report.tar.seq, self.policy.min_tar_seq)
            maximum("gap", report.gap, self.policy.max_gap)
        else:
            for name, expected in (
                ("dar", self.policy.min_dar),
                ("tar_seq", self.policy.min_tar_seq),
                ("gap", self.policy.max_gap),
            ):
                if expected is not None:
                    checks.append(
                        GateCheck(
                            name=name,
                            passed=False,
                            observed="undefined (zero observed groups)",
                            expected=expected,
                        )
                    )
        if self.policy.max_unanimous_path_change_rate is not None:
            if report.observed_groups:
                rate = report.unanimous_with_path_change_rate
                assert rate is not None
                maximum(
                    "unanimous_with_path_change_rate",
                    rate,
                    self.policy.max_unanimous_path_change_rate,
                )
            else:
                checks.append(
                    GateCheck(
                        name="unanimous_with_path_change_rate",
                        passed=False,
                        observed="undefined (zero observed groups)",
                        expected=self.policy.max_unanimous_path_change_rate,
                    )
                )
        if self.policy.max_flags_per_100_cases is not None:
            if report.observed_groups:
                flags_per_100 = report.flags_per_100_cases
                assert flags_per_100 is not None
                maximum(
                    "flags_per_100_cases",
                    flags_per_100,
                    self.policy.max_flags_per_100_cases,
                )
            else:
                checks.append(
                    GateCheck(
                        name="flags_per_100_cases",
                        passed=False,
                        observed="undefined (zero observed groups)",
                        expected=self.policy.max_flags_per_100_cases,
                    )
                )
        if self.policy.max_cost_per_case_usd is not None:
            if report.observed_groups:
                maximum(
                    "cost_per_case_usd",
                    report.total_cost_usd / report.observed_groups,
                    self.policy.max_cost_per_case_usd,
                )
            else:
                checks.append(
                    GateCheck(
                        name="cost_per_case_usd",
                        passed=False,
                        observed="undefined (zero observed groups)",
                        expected=self.policy.max_cost_per_case_usd,
                    )
                )
        minimum(
            "eligible_fraction", report.eligible_fraction, self.policy.min_eligible_fraction
        )
        if self.policy.required_replays is not None:
            checks.append(
                GateCheck(
                    name="replays",
                    passed=report.replays_requested == self.policy.required_replays,
                    observed=float(report.replays_requested),
                    expected=float(self.policy.required_replays),
                )
            )
        if self.policy.min_observed_groups is not None:
            checks.append(
                GateCheck(
                    name="observed_groups",
                    passed=report.observed_groups >= self.policy.min_observed_groups,
                    observed=float(report.observed_groups),
                    expected=f">={self.policy.min_observed_groups}",
                )
            )
        if self.policy.require_complete:
            checks.append(
                GateCheck(
                    name="report_complete",
                    passed=report.status.value == "complete",
                    observed=report.status.value,
                    expected="complete",
                )
            )
        for task_policy in self.policy.by_task:
            rows = [row for row in report.case_reports if row.task == task_policy.task]
            prefix = f"task[{task_policy.task}]"
            checks.append(
                GateCheck(
                    name=f"{prefix}.observed_groups",
                    passed=len(rows) >= task_policy.min_observed_groups,
                    observed=float(len(rows)),
                    expected=f">={task_policy.min_observed_groups}",
                )
            )
            if not rows:
                for suffix, expected in (
                    ("dar", task_policy.min_dar),
                    ("tar_seq", task_policy.min_tar_seq),
                    ("gap", task_policy.max_gap),
                    (
                        "unanimous_with_path_change_rate",
                        task_policy.max_unanimous_path_change_rate,
                    ),
                ):
                    if expected is not None:
                        checks.append(
                            GateCheck(
                                name=f"{prefix}.{suffix}",
                                passed=False,
                                observed="undefined (zero observed groups)",
                                expected=expected,
                            )
                        )
                continue
            task_dar = sum(row.dar for row in rows) / len(rows)
            task_tar = sum(row.tar.seq for row in rows) / len(rows)
            task_gap = task_dar - task_tar
            task_flags = sum(row.unanimous_with_path_change for row in rows) / len(rows)
            minimum(f"{prefix}.dar", task_dar, task_policy.min_dar)
            minimum(f"{prefix}.tar_seq", task_tar, task_policy.min_tar_seq)
            maximum(f"{prefix}.gap", task_gap, task_policy.max_gap)
            maximum(
                f"{prefix}.unanimous_with_path_change_rate",
                task_flags,
                task_policy.max_unanimous_path_change_rate,
            )
        return GateResult(passed=all(check.passed for check in checks), checks=tuple(checks))
