"""Optional OpenTelemetry emission using GenAI semantic conventions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ._meta import package_version
from .exceptions import OptionalDependencyError
from .models import Episode, Manifest, ToolCall


class _NoopSpan:
    def set_attribute(self, name: str, value: Any) -> None:
        return None

    def set_status(self, status: Any) -> None:
        return None


def _set_status(span: Any, *, error_type: str | None = None) -> None:
    """Set an OTel status without recording an exception or sensitive message."""

    try:
        from opentelemetry.trace import Status, StatusCode
    except ImportError:
        return
    if error_type is None:
        span.set_status(Status(StatusCode.OK))
    else:
        span.set_attribute("error.type", error_type)
        span.set_status(Status(StatusCode.ERROR))


def _tracer(enabled: bool) -> Any:
    if not enabled:
        return None
    try:
        from opentelemetry import trace
    except ImportError as exc:
        raise OptionalDependencyError(
            "OpenTelemetry requires `pip install dfah-bench[otel]`"
        ) from exc
    return trace.get_tracer("dfah", package_version())


def preflight(*, enabled: bool = False) -> None:
    """Resolve optional telemetry before any durable dispatch boundary."""

    _tracer(enabled)


@contextmanager
def episode_span(manifest: Manifest, *, enabled: bool = False) -> Iterator[Any]:
    """Create one agent episode span without recording prompts or payloads."""

    tracer = _tracer(enabled)
    if tracer is None:
        yield _NoopSpan()
        return
    attributes = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.provider.name": manifest.provider,
        "gen_ai.request.model": manifest.model,
        "dfah.suite.id": manifest.suite_id,
        "dfah.suite.version": manifest.suite_version,
        "dfah.manifest.hash": manifest.hash,
    }
    with tracer.start_as_current_span(
        "invoke_agent",
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        yield span


def finish_episode_span(span: Any, episode: Episode) -> None:
    """Attach result metadata; raw output and arguments remain excluded."""

    if episode.wire_request is not None:
        span.set_attribute("gen_ai.response.model", episode.wire_request.model)
    span.set_attribute("gen_ai.usage.input_tokens", episode.usage.input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", episode.usage.output_tokens)
    span.set_attribute("dfah.episode.status", episode.status.value)
    span.set_attribute("dfah.parse.strategy", episode.parse.strategy)
    span.set_attribute("dfah.parse.accepted", episode.parse.accepted)
    if episode.decision is not None:
        span.set_attribute("dfah.decision.label", episode.decision.label)
    _set_status(
        span,
        error_type=None if episode.error is None else episode.error.kind,
    )


@contextmanager
def tool_span(tool: ToolCall, *, enabled: bool = False) -> Iterator[Any]:
    """Create one tool span with non-content identity metadata only."""

    tracer = _tracer(enabled)
    if tracer is None:
        yield _NoopSpan()
        return
    attributes: dict[str, str] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": tool.name,
        "gen_ai.tool.type": "function",
    }
    if tool.call_id is not None:
        attributes["gen_ai.tool.call.id"] = tool.call_id
    with tracer.start_as_current_span(
        f"execute_tool {tool.name}",
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        yield span


def finish_tool_span(span: Any, tool: ToolCall) -> None:
    """Attach terminal, non-content tool metadata."""

    span.set_attribute("dfah.tool.execution_state", tool.execution_state.value)
    span.set_attribute("dfah.tool.result_state", tool.result_state.value)
    if tool.latency_ms is not None:
        span.set_attribute("dfah.tool.latency_ms", tool.latency_ms)
    _set_status(
        span,
        error_type=(
            None
            if tool.execution_state.value == "executed"
            else f"dfah.tool.{tool.execution_state.value}"
        ),
    )
