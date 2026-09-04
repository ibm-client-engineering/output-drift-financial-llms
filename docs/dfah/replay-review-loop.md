# Replay, review, and check a correction

This example runs two prewritten local agents against the same two cases and
the same strict gate. It needs no model service or API key. Each episode makes
two actual recorded tool calls: `read_status` and `read_rule`.

From the repository or unpacked source distribution, with DFAH installed:

```bash
python examples/dfah_gate_loop.py --out .dfah/review-loop
```

The first candidate deliberately reverses the tool order on its second replay
of each case. Its final decision stays the same. The corrected candidate always
uses the same order. Both implementations are supplied in the example; the loop
evaluates exactly two candidates and does not generate or edit code.

| Candidate | Decision agreement | Ordered-path agreement | Flagged groups | Gate |
|---|---:|---:|---:|---|
| `01-varying-path` | 1.0 | 0.5 | 2 of 2 | Fail |
| `02-fixed-path` | 1.0 | 1.0 | 0 of 2 | Pass |

The unchanged policy requires complete, artifact-verified evidence, two replays
per case, full eligibility, perfect decision and ordered-path agreement, and
zero decision–path gap or flags. The first candidate therefore fails even
though its decisions agree. The command exits successfully when the supplied
correction passes; the earlier failure remains recorded.

## Review the evidence

The output directory preserves:

- `policy.json`: the policy used for both evaluations;
- `summary.json`: each outcome, failed checks, manifest commitment, and episode
  artifact commitment; and
- one directory per candidate, with its committed episodes, run plan, JSON
  report, `gate.json`, and standalone `report.html`.

Open both HTML reports to compare their case-level paths. Each candidate has a
distinct adapter version, implementation hash, manifest, and run directory.
The suite, cases, tool schemas, and gate stay fixed. Each gate evaluates a report
reloaded and verified against its committed episode store. Running the example
again with the same output path fails before changing existing evidence; choose
a new `--out` for another review.

For a real change, supply the reviewed candidate implementations, keep the
evaluation contract fixed, replay each candidate separately, and inspect its
eligibility and failed checks before accepting the change.

This strict policy treats a change in tool order as a review signal. Here the
two tools are independent, so that signal does not establish an incorrect final
decision. Passing this small example establishes repeatability for its declared
cases and observables; it does not establish general agent correctness.
