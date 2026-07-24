"""Explicit, fixture-friendly tool registration and per-episode recording."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import anyio
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from jsonschema.protocols import Validator

from .._canonical import sha256
from ..exceptions import ConfigurationError, ToolExecutionError
from ..models import (
    ChannelState,
    ToolCall,
    ToolExecutionState,
    Trajectory,
)
from ..suite import ToolSpec
from ..telemetry import finish_tool_span, tool_span

ToolFunction = Callable[..., Any | Awaitable[Any]]


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == () or value == [] or value == {}


class ToolRegistry:
    """An immutable-by-use registry of explicit tool functions.

    Register definitions during setup, then create one :class:`ToolSession` per
    episode. Tool outputs are returned to the agent but only hashes are recorded.
    Raw arguments are also excluded by default while their canonical hash remains
    available to ``TAR_strong``.
    """

    def __init__(self) -> None:
        self._definitions: dict[str, tuple[ToolSpec, ToolFunction, Validator]] = {}
        self._frozen = False

    def register(self, spec: ToolSpec, function: ToolFunction) -> None:
        """Register one named function; duplicate names fail closed."""

        if self._frozen:
            raise ConfigurationError("tool registry is frozen after first use")
        if spec.name in self._definitions:
            raise ConfigurationError(f"tool {spec.name!r} is already registered")
        if not callable(function):
            raise ConfigurationError("tool implementation must be callable")
        schema = spec.model_dump(mode="json")["input_schema"]
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ConfigurationError(f"tool {spec.name!r} has an invalid JSON Schema") from exc
        if schema.get("type") != "object":
            raise ConfigurationError(
                f"tool {spec.name!r} input_schema must declare type=object"
            )
        self._definitions[spec.name] = (
            spec,
            function,
            Draft202012Validator(schema),
        )

    def tool(self, spec: ToolSpec) -> Callable[[ToolFunction], ToolFunction]:
        """Decorator form of :meth:`register`."""

        def decorate(function: ToolFunction) -> ToolFunction:
            self.register(spec, function)
            return function

        return decorate

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """Return schemas in stable name order."""

        return tuple(self._definitions[name][0] for name in sorted(self._definitions))

    @property
    def schema_hash(self) -> str:
        """Canonical hash of the provider-neutral tool contract."""

        return sha256(self.specs)

    def freeze(self) -> ToolRegistry:
        """Prevent schema/function mutation once a replay binds the registry."""

        self._frozen = True
        return self

    def session(
        self,
        *,
        id_prefix: str = "tool",
        capture_arguments: bool = False,
        otel: bool = False,
    ) -> ToolSession:
        """Create isolated recording state for one episode."""

        self.freeze()
        return ToolSession(
            definitions=dict(self._definitions),
            id_prefix=id_prefix,
            capture_arguments=capture_arguments,
            otel=otel,
        )


class ToolSession:
    """Per-episode tool facade that records requested and executed calls."""

    def __init__(
        self,
        *,
        definitions: Mapping[str, tuple[ToolSpec, ToolFunction, Validator]],
        id_prefix: str,
        capture_arguments: bool,
        otel: bool,
    ) -> None:
        self._definitions = dict(definitions)
        self._id_prefix = id_prefix
        self._capture_arguments = capture_arguments
        self._otel = otel
        self._calls: list[ToolCall] = []
        self._lock = anyio.Lock()

    async def _reserve(self, name: str, arguments: Mapping[str, Any]) -> int:
        argument_hash = sha256(dict(arguments))
        async with self._lock:
            index = len(self._calls)
            self._calls.append(
                ToolCall(
                    name=name,
                    arguments=dict(arguments) if self._capture_arguments else {},
                    arguments_hash=argument_hash,
                    arguments_redacted=not self._capture_arguments,
                    call_id=f"{self._id_prefix}-{index:04d}",
                    result_state=ChannelState.UNAVAILABLE,
                    execution_state=ToolExecutionState.REQUESTED,
                )
            )
            return index

    async def call(self, name: str, /, **arguments: Any) -> Any:
        """Invoke one tool and retain a privacy-safe, argument-aware record."""

        index = await self._reserve(name, arguments)
        initial = self._calls[index]
        with tool_span(initial, enabled=self._otel) as span:
            definition = self._definitions.get(name)
            if definition is None:
                async with self._lock:
                    prior = self._calls[index]
                    self._calls[index] = prior.model_copy(
                        update={"execution_state": ToolExecutionState.REJECTED}
                    )
                finish_tool_span(span, self._calls[index])
                raise ToolExecutionError(f"unknown registered tool: {name}")

            _, function, validator = definition
            before = time.perf_counter()
            try:
                validator.validate(dict(arguments))
            except JsonSchemaValidationError as exc:
                location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
                keyword = str(exc.validator or "schema")
                async with self._lock:
                    prior = self._calls[index]
                    self._calls[index] = prior.model_copy(
                        update={
                            "execution_state": ToolExecutionState.REJECTED,
                            "result_state": ChannelState.MALFORMED,
                            "latency_ms": (time.perf_counter() - before) * 1000.0,
                        }
                    )
                finish_tool_span(span, self._calls[index])
                raise ToolExecutionError(
                    f"tool {name!r} arguments violate its JSON Schema at {location} ({keyword})"
                ) from None

            try:
                signature = inspect.signature(function)
                signature.bind(**arguments)
                value = function(**arguments)
                if inspect.isawaitable(value):
                    value = await value
                output_hash = sha256(value)
            except Exception as exc:
                async with self._lock:
                    prior = self._calls[index]
                    self._calls[index] = prior.model_copy(
                        update={
                            "execution_state": ToolExecutionState.ERROR,
                            "result_state": ChannelState.MALFORMED,
                            "latency_ms": (time.perf_counter() - before) * 1000.0,
                        }
                    )
                finish_tool_span(span, self._calls[index])
                if isinstance(exc, ToolExecutionError):
                    raise
                raise ToolExecutionError(f"tool {name!r} failed: {type(exc).__name__}") from exc

            async with self._lock:
                prior = self._calls[index]
                self._calls[index] = prior.model_copy(
                    update={
                        "output_hash": output_hash,
                        "result_state": (
                            ChannelState.OBSERVED_EMPTY
                            if _is_empty(value)
                            else ChannelState.OBSERVED_NONEMPTY
                        ),
                        "execution_state": ToolExecutionState.EXECUTED,
                        "latency_ms": (time.perf_counter() - before) * 1000.0,
                    }
                )
            finish_tool_span(span, self._calls[index])
            return value

    @property
    def calls(self) -> tuple[ToolCall, ...]:
        """Return the recorded calls in request order."""

        return tuple(self._calls)

    def trajectory(self) -> Trajectory:
        """Build a trajectory without treating a zero-tool observation as missing."""

        return Trajectory(
            state=(
                ChannelState.OBSERVED_NONEMPTY if self._calls else ChannelState.OBSERVED_EMPTY
            ),
            tool_calls=tuple(self._calls),
        )
