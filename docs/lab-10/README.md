# Lab 10: Evidence Before Execution

A recorded tool proposal, a completed invocation and a successful business task
answer different questions. This lab uses the public `execution_summary` API
to inspect those boundaries, then checks the shared denominator required for a
small decision/path comparison.

**Duration:** about 15 minutes. **Requirements:** Python 3.10+ and the repository
checkout containing this lab. Installation needs package dependencies; the
example itself uses no model service, API key or network call.

The example constructs **illustrative synthetic records**. No real tool runs,
and none of its numbers are DFAH-Bench study results.

## 1. Install the source checkout

From the repository root:

```bash
python -m venv .venv-dfah
source .venv-dfah/bin/activate
python -m pip install -e .
python -m dfah --version
```

This lab targets the **0.1.3 source tree** and its `execution_summary` API;
it does not assume a published 0.1.3 package or release tag. On Windows, activate
with `.venv-dfah\Scripts\activate`.

## 2. Run the example

```bash
python examples/dfah_execution_evidence.py
```

The command prints JSON to standard output and performs its self-checks. It
writes no run directory. The output identifies the records as
`illustrative_synthetic_records`, reports zero provider calls and `$0.00` in
provider charges, and leaves native task outcomes `not_measured`. Local compute
and energy costs are outside this example.

The main trajectory has six final invocation records:

| Record state | Count | Meaning in this example |
|---|---:|---|
| `EXECUTED` | 3 | A return was captured. |
| `REJECTED` | 1 | The proposal did not reach execution under the capture contract. |
| `REQUESTED` | 1 | No terminal boundary was captured. |
| `ERROR` | 1 | An error was recorded; possible partial effects remain unknown. |

Its completed tool names are `verify_identity`, `transfer_funds`,
`verify_identity`, in that order. Repeated names remain repeated. The original
six records remain in the trajectory; the helper only summarizes them.

One completed transfer record has the synthetic return `Transfer declined.`
It still counts as a completed invocation because a return exists. The helper
does not score the business outcome or decide whether the action followed policy.

`all_proposals_terminal` is false for this mixed trajectory because the
`REQUESTED` record remains unresolved. The separate error-only example returns
true for that property, with zero completed invocations. A terminal error does
not demonstrate rollback or successful execution.

## 3. Keep missing capture distinct from an empty path

The example also summarizes three channel states:

| Channel state | Captured count | Completed tool names | All proposals terminal |
|---|---:|---|---|
| `UNAVAILABLE` | `null` | `null` | `null` |
| `MALFORMED` | `null` | `null` | `null` |
| `OBSERVED_EMPTY` | `0` | `[]` | `true` |

In Python, those unavailable values are `None`, and the observed-empty path is
an empty tuple. The JSON representation uses `null` and `[]`.

An observed-empty channel says that capture recorded no invocations. An
unavailable channel says that the record cannot establish what happened.
Neither a numeric zero nor a completed path should be invented for unavailable
capture. Even an observed channel is bounded by what the adapter captured;
this helper does not verify that every possible effect or call was observed.

## 4. Compare decisions and paths on the same retained records

The final example compares two deliberately observed toy paths with the same
decision, `review`. One path contains two calls and the other one. It reports:

```text
decision_denominator: 2
trajectory_denominator: 2
dar: 1.0
tar_seq: 0.5
gap: 0.5
mismatched_denominators_rejected: true
```

These are arithmetic checks on two supplied records. The example also tries to
pair two decisions with only one path; `delta_dt` rejects that mismatch.
The unavailable and malformed channels above never enter this calculation as
empty paths.

The low-level `dar`, `tar` and `delta_dt` functions expect qualified inputs.
Equal lengths alone do not establish replay eligibility. Real episode groups
also need `evaluate_episode` and `evaluate_group` checks for their required
channels, replay count and comparable manifest/request identities. An execution
summary supplies descriptive counts; it does not replace that qualification.

## 5. Apply the record boundary to your adapter

Pass the final `ToolCall` for each invocation in an ordered `Trajectory`:

```python
from dfah.metrics import execution_summary

summary = execution_summary(episode.trajectory)
print(summary.completed_invocations)
print(summary.rejected_invocations)
print(summary.unresolved_proposals)
print(summary.errored_invocations)
```

If your trace contains separate proposal, verdict and return events for one
invocation, reconcile them into one final invocation record before summarizing.
Keep the original event stream separately. A new retry is a new invocation;
several events for the same invocation are not several completed calls.

This lab illustrates a measurement boundary motivated by v3. It adds no gate
policy, enforcement mechanism, native-benchmark adapter or policy-correctness
score. Assess actual task outcomes and action prerequisites separately using
retained evidence and an explicit task contract.

- [Execution-evidence API guide](../dfah/execution-evidence.md)
- [Lab 8: Replay Measurement](../lab-8/README.md)
- [Lab 9: Replay, Review, and Retest](../lab-9/README.md)
