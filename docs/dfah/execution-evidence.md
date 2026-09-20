# Read execution evidence alongside replay agreement

A safety gate can deny a proposal before its tool implementation runs. A model can also emit a proposal that never reaches dispatch because its agent interface rejects the response. Retaining a tool name does not establish that the operation completed.

`execution_summary` describes the final captured invocation records in an existing `Trajectory`:

```python
from dfah.metrics import execution_summary

summary = execution_summary(episode.trajectory)
print(summary.completed_invocations)
print(summary.rejected_invocations)
print(summary.unresolved_proposals)
print(summary.errored_invocations)
print(summary.completed_tool_names)
```

An unavailable or malformed trajectory returns `None` for its counts and completed path. An observed empty trajectory returns zero counts and an empty tuple. Repeated names retain their order and multiplicity.

A completed invocation means the adapter recorded an `EXECUTED` call with a captured return. It does not mean the business task succeeded, the action followed policy, or all effects were captured. A tool can return a failure message normally. An `ERROR` can happen after partial effects; the summary does not assume rollback. `all_invocations_resolved` says whether any captured record remains `REQUESTED`, not whether all calls succeeded or all possible calls were observed.

The input is the final `ToolCall` per invocation, as retained by DFAH's standard tool session. Do not concatenate request, verdict and return events for the same invocation into this input. If an external trace is an event stream, reconcile its events into final invocation records first and preserve the original stream separately.

This helper does not modify historical DAR, TAR, report schemas or eligibility. Use it to explain what the retained path contains. Policy correctness still needs an independently specified task or action contract; repeated agreement alone does not provide that contract.

The addition is motivated by the DFAH-Bench v3 external-benchmark integration, where proposals, gate verdicts, dispatches, returns, database effects and capability changes are retained separately. Those research traces include more channels than the generic `ToolCall` record. This summary does not certify coverage of those additional effects.
