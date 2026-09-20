"""Regression: denied/undispatched writes must not appear as completed actions."""

import pytest

from dfah.metrics.execution import execution_summary
from dfah.models import ChannelState, ToolCall, ToolExecutionState, Trajectory


def call(name, state):
    return ToolCall(
        name=name,
        execution_state=state,
        result_state=ChannelState.OBSERVED_NONEMPTY
        if state is ToolExecutionState.EXECUTED
        else ChannelState.UNAVAILABLE,
        output_hash="a" * 64 if state is ToolExecutionState.EXECUTED else None,
    )


def test_gate_denial_then_recovery_preserves_only_completed_return_order():
    path = Trajectory(
        state=ChannelState.OBSERVED_NONEMPTY,
        tool_calls=(
            call("change_email", ToolExecutionState.REJECTED),
            call("log_verification", ToolExecutionState.EXECUTED),
            call("change_email", ToolExecutionState.EXECUTED),
            call("change_email", ToolExecutionState.REQUESTED),
        ),
    )
    result = execution_summary(path)
    assert result.captured_invocations == 4
    assert result.completed_invocations == 2
    assert result.rejected_invocations == 1
    assert result.unresolved_proposals == 1
    assert result.completed_tool_names == ("log_verification", "change_email")
    assert result.all_invocations_resolved is False
    assert len(path.tool_calls) == 4


@pytest.mark.parametrize("state", [ChannelState.UNAVAILABLE, ChannelState.MALFORMED])
def test_unavailable_never_becomes_observed_zero(state):
    result = execution_summary(Trajectory(state=state))
    assert result.captured_invocations is None
    assert result.completed_tool_names is None
    assert result.all_invocations_resolved is None


def test_observed_empty_is_distinct_and_errors_do_not_become_success():
    empty = execution_summary(Trajectory(state=ChannelState.OBSERVED_EMPTY))
    assert empty.captured_invocations == 0 and empty.completed_tool_names == ()
    assert empty.all_invocations_resolved is True
    error = execution_summary(
        Trajectory(
            state=ChannelState.OBSERVED_NONEMPTY,
            tool_calls=(call("transfer_money", ToolExecutionState.ERROR),),
        )
    )
    assert error.errored_invocations == 1 and error.completed_invocations == 0
    assert error.all_invocations_resolved is True


def test_names_keep_multiplicity_and_require_validated_channel():
    result = execution_summary(
        Trajectory(
            state=ChannelState.OBSERVED_NONEMPTY,
            tool_calls=(
                call("verify", ToolExecutionState.EXECUTED),
                call("verify", ToolExecutionState.EXECUTED),
            ),
        )
    )
    assert result.completed_tool_names == ("verify", "verify")
    with pytest.raises(TypeError):
        execution_summary([])
