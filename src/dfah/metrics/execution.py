"""Descriptive execution boundaries, separate from replay agreement and safety.

A rejected or unresolved proposal is not a successfully completed invocation.
An error may occur after a tool changed state. These summaries therefore make
no claim about rollback, business effects, policy correctness or tool coverage
outside the supplied observed channel.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..models import ChannelState, ToolExecutionState, Trajectory


@dataclass(frozen=True)
class ExecutionSummary:
    """Counts are unavailable when the source channel is unavailable/malformed.

    Counts describe retained invocation records, not an append-only event stream.
    Pass the final ToolCall per invocation, as a standard Trajectory does; do not
    concatenate proposed, rejected and executed events for the same invocation.
    Names preserve order and multiplicity; they are not a new TAR metric.
    """

    source_state: ChannelState
    captured_invocations: int | None
    completed_invocations: int | None
    rejected_invocations: int | None
    unresolved_proposals: int | None
    errored_invocations: int | None
    completed_tool_names: tuple[str, ...] | None

    @property
    def all_invocations_resolved(self) -> bool | None:
        """Whether every captured record has a terminal boundary, including error.

        Terminal does not imply success, a captured effect, or absence of harm.
        """
        if not self.source_state.observed:
            return None
        return self.unresolved_proposals == 0


def execution_summary(trajectory: Trajectory) -> ExecutionSummary:
    """Summarize the retained dispatch boundaries without changing DAR or TAR.

    A call marked EXECUTED has a captured return under DFAH's ToolCall contract;
    this does not certify that the return denotes business success. ERROR keeps
    possible partial effects unresolved. REQUESTED has no terminal evidence;
    REJECTED did not reach execution under the adapter's capture contract.
    """
    if not isinstance(trajectory, Trajectory):
        raise TypeError("execution_summary requires a validated Trajectory")
    if not trajectory.state.observed:
        return ExecutionSummary(trajectory.state, None, None, None, None, None, None)
    counts = Counter(call.execution_state for call in trajectory.tool_calls)
    return ExecutionSummary(
        source_state=trajectory.state,
        captured_invocations=len(trajectory.tool_calls),
        completed_invocations=counts[ToolExecutionState.EXECUTED],
        rejected_invocations=counts[ToolExecutionState.REJECTED],
        unresolved_proposals=counts[ToolExecutionState.REQUESTED],
        errored_invocations=counts[ToolExecutionState.ERROR],
        completed_tool_names=tuple(
            call.name
            for call in trajectory.tool_calls
            if call.execution_state is ToolExecutionState.EXECUTED
        ),
    )
