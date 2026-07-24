"""Versioned replay suites and built-in synthetic examples."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import Field, model_validator

from ._canonical import sha256
from ._frozen import FrozenJsonMap
from .exceptions import ConfigurationError
from .models import SEMVER_PATTERN, Case, Record


class ToolSpec(Record):
    """Provider-neutral JSON-schema description of one deterministic tool."""

    name: str = Field(min_length=1)
    description: str = ""
    input_schema: FrozenJsonMap = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_object_schema(self) -> ToolSpec:
        schema = self.model_dump(mode="json")["input_schema"]
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ValueError(
                "tool input_schema is not valid Draft 2020-12 JSON Schema"
            ) from exc
        if schema.get("type") != "object":
            raise ValueError("tool input_schema must declare type=object")
        return self

    def validate_arguments(self, arguments: Mapping[str, Any]) -> None:
        """Validate one argument object without echoing rejected values."""

        validator = Draft202012Validator(self.model_dump(mode="json")["input_schema"])
        errors = sorted(
            validator.iter_errors(dict(arguments)), key=lambda error: list(error.path)
        )
        if not errors:
            return
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        keyword = str(error.validator or "schema")
        raise ValueError(
            f"tool arguments violate the declared JSON Schema at {location} ({keyword})"
        )


class Suite(Record):
    """A named, semantically versioned set of replay cases.

    Fixture hashes detect byte-level drift. ``suite_version`` tells a human
    whether the behavioral contract changed.
    """

    suite_id: str = Field(min_length=1)
    suite_version: str = Field(pattern=SEMVER_PATTERN)
    decisions: tuple[str, ...] = Field(min_length=2)
    cases: tuple[Case, ...] = Field(min_length=1)
    tools: tuple[ToolSpec, ...] = ()
    required_channels: tuple[Literal["decision", "trajectory", "evidence"], ...] = (
        "decision",
        "trajectory",
    )
    description: str = ""

    @model_validator(mode="after")
    def _unique_contract(self) -> Suite:
        normalized = [label.strip().lower() for label in self.decisions]
        if any(not label for label in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("suite decisions must be unique nonempty labels")
        if any(re.fullmatch(r"[a-z][a-z0-9_]*", label) is None for label in normalized):
            raise ValueError(
                "suite decisions must use parser-compatible labels: letters, digits, "
                "and underscores"
            )
        if tuple(sorted(self.required_channels)) != ("decision", "trajectory"):
            raise ValueError(
                "current DFAH suites require exactly decision and trajectory; evidence is "
                "not yet captured by Episode"
            )
        case_ids = [case.effective_case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("suite artifact case IDs must be unique")
        tool_names = [tool.name for tool in self.tools]
        if len(set(tool_names)) != len(tool_names):
            raise ValueError("suite tool names must be unique")
        return self

    @property
    def fixture_hash(self) -> str:
        """Hash of all case content and suite semantics."""

        return sha256(
            {
                "suite_id": self.suite_id,
                "suite_version": self.suite_version,
                "decisions": self.decisions,
                "cases": self.cases,
                "required_channels": self.required_channels,
            }
        )

    @property
    def tool_schema_hash(self) -> str:
        """Hash of the public tool contract."""

        return sha256(tuple(sorted(self.tools, key=lambda tool: tool.name)))

    @classmethod
    def load(cls, source: Suite | str | Path) -> Suite:
        """Load a suite object, a built-in suite name, or JSON/YAML file."""

        if isinstance(source, Suite):
            return source
        if isinstance(source, Path) or Path(str(source)).is_file():
            path = Path(source)
            if path.suffix.lower() == ".json":
                return cls.model_validate_json(path.read_bytes())
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise ConfigurationError("YAML suite loading requires PyYAML") from exc
            return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        try:
            return _BUILTIN_SUITES[str(source)]
        except KeyError as exc:
            names = ", ".join(sorted(_BUILTIN_SUITES))
            raise ConfigurationError(f"unknown suite {source!r}; built-ins: {names}") from exc


_COMPLIANCE_V1 = Suite(
    suite_id="compliance-v1",
    suite_version="1.0.0",
    decisions=("escalate", "dismiss", "investigate"),
    cases=(
        Case(
            case_id="SYN-TXN-001",
            task="compliance",
            input={"alert_type": "sanctions_name_match", "risk_tier": "high"},
        ),
        Case(
            case_id="SYN-TXN-002",
            task="compliance",
            input={"alert_type": "duplicate_low_value", "risk_tier": "low"},
        ),
    ),
    description=(
        "Tiny synthetic integration suite. It validates plumbing; it is not a financial "
        "accuracy benchmark."
    ),
)

_CONFORMANCE_V1 = Suite(
    suite_id="conformance-v1",
    suite_version="1.0.0",
    decisions=("pass", "review", "reject"),
    cases=(
        Case(case_id="CONF-001", task="conformance", input={"value": 7, "limit": 10}),
        Case(case_id="CONF-002", task="conformance", input={"value": 12, "limit": 10}),
    ),
    description="Built-in no-network suite used by dfah.testing.check_agent.",
)

_BUILTIN_SUITES = {
    _COMPLIANCE_V1.suite_id: _COMPLIANCE_V1,
    _CONFORMANCE_V1.suite_id: _CONFORMANCE_V1,
}


def list_suites() -> tuple[Suite, ...]:
    """Return built-in suites in stable name order."""

    return tuple(_BUILTIN_SUITES[name] for name in sorted(_BUILTIN_SUITES))
