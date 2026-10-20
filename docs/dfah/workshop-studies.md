# Replay, capture, and authorization

When an agent repeats its answer, what do we know about the work behind it?
The DFAH studies examine three parts of that question: whether the path
repeats, whether the record preserves the path, and whether an action has
the evidence it needs.

A successful rerun can leave a better final answer without showing the earlier
attempts that led to it. If retries overwrite records, the final state may no
longer reveal an earlier action or access. That engineering question motivates
preserving execution evidence; the studies below test specific replay and
capture failures, not the effects of script revision or context compression.

Two workshop papers develop different parts of this research:

| Study | Presentation | Question |
|---|---|---|
| [The Determinism–Faithfulness Assurance Harness (DFAH): Evidence Before Execution in Financial Agent Gates](https://openreview.net/forum?id=VWQqIgrKw4) | Accepted for a poster at AI4F, ACM ICAIF 2026 | Does a repeated authorization decision have supporting evidence, and what happens to task completion and cost? |
| [When Replay Agreement Misleads: Capture Loss and Cancellation in Local Agents](https://openreview.net/forum?id=njEKVxonV2) | Accepted for a poster at SLM-Agents, NeurIPS 2026 | Can an average hide a changed path, or repeated records hide the same missing calls? |

The [DFAH-Bench preprint](https://arxiv.org/abs/2607.20491) connects these
questions. The papers share some observations; their counts are not
independent experiments to add together.

## SLM-Agents: agreement can survive missing evidence

In the historical Qwen example, two case-level gaps of +0.125 and −0.125
cancel in the average. Yet one of 97 unanimous-decision groups changes its
tool path. Those 800 episodes already belong to the historical replay corpus.

The separate capture diagnostic contains **48 executions in 16 complete
groups**. Removing the first call, or every call, from each repeated record
leaves agreement at **1.0**. All 48 records in each deletion condition have
an omission. Comparing them with the retained dispatcher observations detects
the omissions; repeated agreement alone does not. The three record views make
144 views, not 144 model executions.

### Inspect the released supplement

Download the [SLM-Agents supplementary ZIP](https://openreview.net/attachment?id=njEKVxonV2&name=supplementary_material)
from OpenReview and unpack it into an empty directory. From that directory:

```bash
python3 scripts/verify_historical_local.py
python3 scripts/verify_local_capture.py
python3 -m unittest discover -s scripts -p 'test_*.py' -v
```

These commands use Python 3.11's standard library, make no model calls, and
produce two PASS reports and 13 passing tests. The capture verifier rewrites
its derived audit JSON. The historical verifier checks 400 derived case groups;
it does not reconstruct the original historical responses.

The supplement contains the frozen capture plan, synthetic episode projections,
collector, tests, and offline verifiers. It is a separate distribution with its
own rights statement. The repository does not mirror or relicense it.

## AI4F: a repeated verdict is only part of the result

The banking study scheduled **1,080 episodes**, with **1,033 known outcomes**
and 47 unknown outcomes across separate cohorts. In the primary schedule,
structural checks alone completed more tasks than either model-based
gate-and-recovery bundle. Input limits and recovery behavior are part of
those comparisons.

The typed-choice bundle had lower mean episode cost on complete task pairs
than the generative bundle, but fewer successes across the primary schedule
under every assignment of unknown outcomes. Missing outcomes prevented the
planned confirmatory tests, so these comparisons are descriptive. The
[banking study guide](v3-study.md) gives the counts and comparison populations.

Fixed-state probes ask whether verdicts match constructed rules. They keep
rule match separate from repeated agreement and from the banking benchmark's
task-success measure.

[Lab 10](../lab-10/README.md) teaches the related distinction between a proposed
call, a returned invocation, and a completed task using synthetic package
records. The hosted banking runner and its study records are not distributed
here; the lab does not reproduce those experiments.

## Choose the material for your question

| Goal | Start here |
|---|---|
| Measure decision and recorded-path agreement | [Lab 8](../lab-8/README.md) |
| Inspect a path change and apply a fixed review policy | [Lab 9](../lab-9/README.md) |
| Separate proposed, rejected, completed, and unresolved invocations | [Lab 10](../lab-10/README.md) |
| Recompute the capture diagnostic | The OpenReview supplement and commands above |
| Reproduce the historical replay analysis | `make reproduce-paper` from this repository |
| Read the banking methods and findings | The AI4F paper and banking study guide |
