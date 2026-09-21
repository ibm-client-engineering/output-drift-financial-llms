"""Illustrative execution records: no provider calls, tools, or benchmark data.

Run from a checkout installed with `python -m pip install -e .`:
    python examples/dfah_execution_evidence.py

These are constructed final invocation records, not an append-only event log.
The self-checks exercise public APIs without adding an enforcement policy.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256

from dfah import ChannelState, ToolCall, ToolExecutionState, Trajectory
from dfah.metrics import dar, delta_dt, execution_summary, tar


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def completed(name: str, call_id: str, captured_return: str) -> ToolCall:
    """Give a synthetic nonempty return its exact UTF-8 content hash."""
    return ToolCall(
        name=name,
        call_id=call_id,
        execution_state=ToolExecutionState.EXECUTED,
        result_state=ChannelState.OBSERVED_NONEMPTY,
        output_hash=sha256(captured_return.encode("utf-8")).hexdigest(),
    )


def summarize(trajectory: Trajectory) -> dict[str, object]:
    result = execution_summary(trajectory)
    return {**asdict(result), "all_proposals_terminal": result.all_proposals_terminal}


def main() -> None:
    # Each call_id names one final invocation record. A later retry is another
    # invocation, not another event for the same invocation.
    calls = (
        ToolCall(
            name="transfer_funds",
            call_id="inv-01",
            execution_state=ToolExecutionState.REJECTED,
        ),
        completed("verify_identity", "inv-02", "Identity check returned."),
        completed("transfer_funds", "inv-03", "Transfer declined."),
        completed("verify_identity", "inv-04", "Identity check returned again."),
        ToolCall(
            name="transfer_funds",
            call_id="inv-05",
            execution_state=ToolExecutionState.REQUESTED,
        ),
        ToolCall(
            name="transfer_funds",
            call_id="inv-06",
            execution_state=ToolExecutionState.ERROR,
        ),
    )
    mixed = Trajectory(state=ChannelState.OBSERVED_NONEMPTY, tool_calls=calls)
    result = execution_summary(mixed)
    check(len({call.call_id for call in calls}) == len(calls), "Use unique invocation IDs")
    check(result.captured_invocations == 6, "All captured invocation records must remain")
    check(
        (
            result.completed_invocations,
            result.rejected_invocations,
            result.unresolved_proposals,
            result.errored_invocations,
        )
        == (3, 1, 1, 1),
        "Completed, rejected, unresolved and errored records must stay distinct",
    )
    check(
        result.completed_tool_names == ("verify_identity", "transfer_funds", "verify_identity"),
        "The completed path must preserve order and repeated names",
    )
    check(result.all_proposals_terminal is False, "The requested record is unresolved")
    check(len(mixed.tool_calls) == 6, "Summarizing must not filter the original trajectory")
    # A normally returned refusal is still a completed invocation. There is no
    # task-success or policy-compliance assessment in execution_summary.
    declined = execution_summary(
        Trajectory(state=ChannelState.OBSERVED_NONEMPTY, tool_calls=(calls[2],))
    )
    check(declined.completed_invocations == 1, "Completion records a return, not its meaning")

    unavailable = Trajectory(state=ChannelState.UNAVAILABLE)
    malformed = Trajectory(state=ChannelState.MALFORMED)
    empty = Trajectory(state=ChannelState.OBSERVED_EMPTY)
    for path in (unavailable, malformed):
        summary = execution_summary(path)
        check(summary.captured_invocations is None, "Unknown capture must not become zero")
        check(
            summary.completed_tool_names is None,
            "Unknown capture must not become an empty path",
        )
        check(summary.all_proposals_terminal is None, "Unknown capture must stay unknown")
    empty_summary = execution_summary(empty)
    check(empty_summary.captured_invocations == 0, "Observed-empty capture has zero records")
    check(empty_summary.completed_tool_names == (), "Observed-empty capture has an empty path")
    check(
        empty_summary.all_proposals_terminal is True, "No captured proposal remains unresolved"
    )

    error_only = Trajectory(state=ChannelState.OBSERVED_NONEMPTY, tool_calls=(calls[-1],))
    error_summary = execution_summary(error_only)
    check(error_summary.errored_invocations == 1, "An error remains an error")
    check(error_summary.completed_invocations == 0, "An error is not a completed return")
    check(error_summary.all_proposals_terminal is True, "Error is a terminal record")

    # Low-level agreement metrics take already-qualified inputs. These two
    # deliberately observed toy paths share the same two-record denominator.
    # Missing channels above never enter this comparison as empty paths.
    paths = (
        Trajectory(state=ChannelState.OBSERVED_NONEMPTY, tool_calls=(calls[1], calls[2])),
        Trajectory(state=ChannelState.OBSERVED_NONEMPTY, tool_calls=(calls[1],)),
    )
    decisions = ("review", "review")
    check(all(path.state.observed for path in paths), "Toy comparison requires observed paths")
    check(dar(decisions) == 1.0 and tar(paths) == 0.5, "The toy decision/path contrast changed")
    check(delta_dt(decisions, paths) == 0.5, "The paired toy gap changed")
    try:
        delta_dt(decisions, paths[:1])
    except ValueError as error:
        check("identical episode denominator" in str(error), "Unexpected denominator failure")
    else:
        raise AssertionError("Mismatched decision/path denominators must be rejected")

    print(
        json.dumps(
            {
                "record_kind": "illustrative_synthetic_records",
                "provider_calls": 0,
                "provider_cost_usd": "0.00",
                "native_task_outcomes": "not_measured",
                "mixed_invocations": summarize(mixed),
                "unavailable_capture": summarize(unavailable),
                "malformed_capture": summarize(malformed),
                "observed_empty_capture": summarize(empty),
                "error_only": summarize(error_only),
                "illustrative_paired_comparison": {
                    "decision_denominator": len(decisions),
                    "trajectory_denominator": len(paths),
                    "dar": dar(decisions),
                    "tar_seq": tar(paths),
                    "gap": delta_dt(decisions, paths),
                    "mismatched_denominators_rejected": True,
                },
                "self_checks": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
