"""Small provider-neutral agent protocol used by the replay orchestrator."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload, runtime_checkable

from pydantic import Field

from ..exceptions import AgentContractError
from ..models import Case, Manifest, Record, Trajectory, Usage, WireRequest

if TYPE_CHECKING:
    from ..suite import Suite
    from ..tools import ToolRegistry, ToolSession


class AgentResult(Record):
    """Provider-neutral outcome returned before DFAH parses the final decision."""

    output_text: str
    trajectory: Trajectory
    wire_request: WireRequest
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = Field(default=0.0, ge=0.0)
    provider_response_id: str | None = None
    raw_capture_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RunContext:
    """Explicit execution identity supplied to a user agent."""

    manifest_hash: str
    case_id: str
    replay_index: int
    seed: int | None
    idempotency_key: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tools: ToolSession | None = None


@runtime_checkable
class Agent(Protocol):
    """One async method plus a pinned manifest—no framework dependency."""

    @property
    def manifest(self) -> Manifest:
        """Return the request contract this adapter will echo after dispatch."""

    async def arun(self, case: Case, context: RunContext) -> AgentResult:
        """Execute one episode without performing hidden replay retries."""


AgentCallable = Callable[[Case, RunContext], AgentResult | Awaitable[AgentResult]]


class CallableAgent:
    """Adapt an ordinary typed callable to the :class:`Agent` protocol."""

    def __init__(
        self,
        function: AgentCallable,
        manifest: Manifest,
        tools: ToolRegistry | None = None,
        suite: Suite | None = None,
    ) -> None:
        if not callable(function):
            raise AgentContractError("agent function must be callable")
        self._function = function
        self._manifest = manifest
        self._tools = tools
        self._suite = suite

    @property
    def manifest(self) -> Manifest:
        """Pinned adapter manifest."""

        return self._manifest

    @property
    def tools(self) -> ToolRegistry | None:
        """Optional registry injected into every replay context."""

        return self._tools

    @property
    def suite(self) -> Suite | None:
        """Optional suite binding used by the one-line conformance check."""

        return self._suite

    async def arun(self, case: Case, context: RunContext) -> AgentResult:
        """Call and validate either a sync or async user function."""

        value = self._function(case, context)
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, AgentResult):
            raise AgentContractError(
                "agent must return dfah.AgentResult; implicit dict/string coercion is disabled"
            )
        return value


_F = TypeVar("_F", bound=AgentCallable)


@overload
def agent(
    *,
    manifest: Manifest,
    tools: ToolRegistry | None = None,
    suite: Suite | None = None,
) -> Callable[[_F], CallableAgent]: ...


@overload
def agent(
    function: _F,
    *,
    manifest: Manifest,
    tools: ToolRegistry | None = None,
    suite: Suite | None = None,
) -> CallableAgent: ...


def agent(
    function: _F | None = None,
    *,
    manifest: Manifest,
    tools: ToolRegistry | None = None,
    suite: Suite | None = None,
) -> CallableAgent | Callable[[_F], CallableAgent]:
    """Wrap a typed callable as a DFAH agent.

    Example:
        ``@agent(manifest=my_manifest)`` followed by an async function accepting
        ``(case, context)`` and returning :class:`AgentResult`.
    """

    def wrap(candidate: _F) -> CallableAgent:
        return CallableAgent(candidate, manifest, tools, suite)

    return wrap(function) if function is not None else wrap
