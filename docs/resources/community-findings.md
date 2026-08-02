# Community reports and findings

_Last updated: August 2, 2026_

This page is a dated ledger of contributor-reported experiments, public defect
reports, and review themes that sharpened the interpretation of the work.
Entries are not part of the corrected DFAH-Bench v2 evidence set unless the
benchmark and reproduction materials explicitly say otherwise. Review themes
are summarized without attribution or quotations.

## Contributor-reported experiment — November 17, 2025

Paul Merrison, whose FINOS affiliation was recorded in the original note,
reported a bounded experiment described as covering six models from 3B to
20B. Four results were preserved in that note:

| Configuration label as reported | Reported output agreement |
| --- | ---: |
| Qwen2.5-7B | 100% |
| Gemma2-9B | 100% |
| Llama3.1-8B | 62.5% |
| Mistral-7B | 33% RAG; 100% SQL |

The other two results, repository commit, exact model revisions,
provider/runtime, prompts and corpus, decoding settings, replay counts and
denominators, exclusions, and raw outputs were not preserved in the public
note. The observations are therefore contributor-reported and are not
directly comparable with current DFAH decision or trajectory metrics.

The preserved results show variation across the recorded model/task
combinations. They do not establish a model-size law or isolate architecture,
training, provider, task, or harness effects. An affiliation records context;
it does not imply organizational endorsement.

## Review themes — July 2026

Community review raised several useful interpretation questions. The themes
below are synthesized without attribution.

### Why use finance if the measurement is domain-agnostic?

Finance is a useful first testbed because two runs can reach the same decision
while executing different controls, accessing different data, or taking
different actions. The measurement design can be used in other domains, but
the current validation remains bounded to synthetic financial workflows.
Adoption elsewhere requires a reviewed, domain-specific suite and separate
correctness, safety, and policy evaluation.

### Do the provider-default Claude runs invalidate those results?

The missing explicit temperature invalidates the historical claim that those
Claude runs used temperature 0.0 and prevents a clean, controlled provider
comparison. It does not erase the observed runs. DFAH-Bench v2 labels them as
provider-default configurations and treats them as bounded sensitivity
evidence. The prospective diagnostic separately captures arguments and
deterministic result identities under a frozen protocol, with its coverage
limitation disclosed.

### Should a team optimize a model toward one DFAH threshold?

DFAH is better treated as a diagnostic and regression control around the
deployed system—model, prompt, tools, harness, and provider settings—than as a
model-training objective. There is no universal passing threshold. A team
should calibrate its own policy against workflow risk, baseline behavior,
review load, latency, and cost, while evaluating correctness and compliance
separately.

### Is a repeatable tool path always desirable?

No. Path diversity can be useful in exploratory or idea-generation work, and
a difference is not automatically a defect. Path stability becomes more
important when tools enforce controls, access sensitive data, change state, or
create side effects. If a step must be exactly deterministic, a reusable
script or direct unit test is usually the stronger control.

Consistent capability selection across different prompts and replay stability
under an identical case/configuration answer complementary questions. The
first belongs in a representative golden-set evaluation; DFAH focuses on what
still changes after the replay contract is held fixed.

## Public defect report and fix — July 20 to August 2, 2026

On July 20, [issue #2](https://github.com/ibm-client-engineering/output-drift-financial-llms/issues/2)
reported that two legacy workshop validators could fail open:

- absent outputs such as `[None, None]` could be reported as perfectly
  consistent; and
- an unknown task type could select no requirements and be reported as
  compliant.

Maintainer review reproduced both behaviors and found a third: a recognized
task with every required profile marked `not_evaluated` could still receive a
successful aggregate verdict.

On August 2, [PR #12](https://github.com/ibm-client-engineering/output-drift-financial-llms/pull/12)
fixed all three cases. Blank, non-string, or otherwise invalid output
observations no longer pass; unknown task types fail closed; and missing
required evaluations produce an incomplete result while retaining the
per-requirement `not_evaluated` detail. Regression coverage lives in
`tests/test_regulatory_invariants.py`, and both the pull-request and
post-merge research reproduction workflows passed.

This hardening applies to the historical workshop validators. It does not
retroactively reproduce or validate the November 2025 contributor report, and
it does not turn a consistency result into a correctness or compliance
attestation.

## Submit a reproducible community report

Open an [issue](https://github.com/ibm-client-engineering/output-drift-financial-llms/issues/new)
or [pull request](https://github.com/ibm-client-engineering/output-drift-financial-llms/pulls)
with as many of these as can be shared safely:

- repository commit;
- exact model and provider identifiers;
- task, prompt, and corpus versions;
- decoding and request settings;
- replay count, denominator, and exclusions;
- metric definition and commands;
- environment details; and
- a public or sanitized result artifact.

Do not include credentials, confidential prompts, customer data, or raw
provider captures that are not approved for release. A report remains
“contributor-reported” until another party can reproduce it from the published
materials.
