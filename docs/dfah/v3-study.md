# V3: evidence, execution, outcomes and cost

A repeated decision can still lack a required fact. DFAH-Bench v3 carries the
replay study's qualification principle into interactive banking tasks: retain
what the agent saw, what a check permitted, what actually ran and whether the
task finished successfully. Each measure keeps its own unit and denominator.

The [manuscript and build instructions](https://github.com/ibm-client-engineering/output-drift-financial-llms/tree/main/paper/arxiv_dfah_bench_v3)
contain the full methods and figures. The paper remains part of the
[DFAH-Bench record, arXiv:2607.20491](https://arxiv.org/abs/2607.20491).
The source revision and the version currently served by arXiv can differ.

## What the extension measures

| Measurement | Unit and question |
|---|---|
| Replay agreement | Qualified identical-input replay groups: does the decision or recorded path repeat? |
| Decision-boundary evidence | A proposed action: which policy, dialogue and tool results reached the gate? |
| Execution evidence | A captured invocation: was it rejected, unresolved, returned or errored? |
| Native task outcome | A scheduled interactive episode: did the upstream evaluator score completion? |
| Fixed-state gate probes | Repeated judgments on selected contexts: does the choice repeat and match a constructed policy label? |
| Economics | Qualified matched tasks: how do all-role API costs and native success change together? |

The native environment is Sierra's **τ-Knowledge banking_knowledge**, distributed
in the τ³-bench codebase under the `tau2-bench` repository name. The manuscript
pins its source and credits the benchmark separately from this intervention.
The three arms share structural checks. A uses those checks alone, B adds a
prose-model allow/block/review decision and C uses the Jev typed-choice service.
The comparison includes admission limits and the recovery allowance as well
as the gate's model response.

## Reading the completed results

All **1,080 scheduled native episodes** reached a terminal artifact;
**1,033 have known outcomes and 47 remain unknown**. The primary cohort uses
the open-weight DeepSeek v4.1 Flash generator. A separate Gemini 3.8 Flash
generator cohort provides a frontier-model replication, and a separate
DeepSeek BM25 cohort changes retrieval. Gemini also supplies simulation and
grading. The cohorts retain separate comparisons.

For the primary 810-episode schedule (270 episodes per arm):

| Arm | Known successes | Known outcomes | Unknown outcomes |
|---|---:|---:|---:|
| A: structural checks | 121 | 260 | 10 |
| B: prose-model gate | 81 | 263 | 7 |
| C: typed-choice gate | 70 | 263 | 7 |

On the 77 tasks with all B and C repeats observed, C minus B native success
is −3.03 percentage points, with a descriptive task-bootstrap interval of
[−7.79, +1.73]. Across all 90 scheduled tasks, assigning every unknown outcome
both possible values bounds the difference to [−6.67, −1.48] points. These
are different quantities: the first describes the complete-task subset; the
second bounds this finite schedule. The planned complete-cohort confirmatory
test was **unavailable**, and its secondary gate remained closed.

The fixed-state probes ask another question. In the constructed synthetic
bank, C had higher observed decision agreement (98.7% versus 93.1%) and lower
policy-label match (54.5% versus 82.6%). The selected referral case also showed
that stable permission can coexist with missing current-time evidence. Those
labels and selected contexts have their own validation limits; they are not
native task-success measurements or a general model ranking.

## Cost belongs beside the outcome

On the same 77 complete primary task pairs, C's mean all-role episode cost was
$0.004345 lower than B's. That reduction coexisted with fewer successes under
every completion of the scheduled unknown outcomes. Input admission and
recovery exhaustion materially affected the observed bundles, so a cheaper
episode alone does not establish a better intervention.

OpenRouter routed generator, simulator, native LLM-grader and B-gate requests
through its chat-completions endpoint; Jev used its SystemOne endpoint.
Requested and returned model/provider identities, request settings, accounting
bases and available cache metadata were retained. Hosted inference is distinct
from the earlier local Ollama experiments and the paper's illustrative local
hardware sizing discussion.

The three native cohorts used **$27.25** in reconciled API cost. For the broader
study, a September 21, 2026 key snapshot reported **$38.97** combined credit
and BYOK upstream usage; the conservative ledger recorded **$39.08** against
the $80 ceiling. These totals cover different scopes from the paired episode
comparison. The manuscript explains the small review allocations and incomplete
cost traces. Later editorial model review is outside those study totals.

## What you can reproduce from this repository

| Surface | Local path or command | What it reproduces |
|---|---|---|
| Corrected v2 analysis | `make reproduce-paper` | Preserved retrospective results and aggregate checks from the public fixture |
| V3 paper | Build `paper/arxiv_dfah_bench_v3/main.tex` | Manuscript presentation from included sources and figure PDFs |
| Package behavior | `python -m pytest tests/dfah` | Capture, eligibility, replay, export and execution-summary contracts |
| Offline teaching example | [Lab 10](../lab-10/README.md) | Synthetic invocation summaries and shared-denominator behavior |

The public package and lab do not include the hosted τ collection runner or
the retained provider logs needed to regenerate its new analyses. The paper
documents those access and runtime limits. Re-running a hosted model today is
a new experiment; it cannot refreeze the original service. Compiling the paper
or running the synthetic lab therefore does not regenerate its empirical
findings.

Start with [Lab 8](../lab-8/README.md) for replay eligibility,
[Lab 9](../lab-9/README.md) for a fixed-policy review loop, and
[Lab 10](../lab-10/README.md) for the execution boundary.
