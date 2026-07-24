# Lab 8: DFAH-Bench — Replay Measurement

## Overview

This lab introduces **DFAH-Bench**, the replay benchmark behind
*DFAH-Bench: Benchmarking Observable Agent Instability in Financial
Decision-Making*
([arXiv:2607.20491](https://arxiv.org/abs/2607.20491)).
It asks a practical question: when repeated agent runs reach the same decision,
did they also follow the same recorded tool path?

The corrected primary analysis contains **4,157 episodes from configurations
with observed tool use across 719 comparable groups**, eight configurations,
and two synthetic financial tasks. Among 627 unanimous-decision groups, 122
(19.5%) changed tool sequence and 47 (7.5%) changed the tool-name set. These
are replay and observability results, not claims about correctness or safety.

**Duration**: about 30 minutes. The package track is local and makes no provider
calls.

## Learning objectives

By the end of this lab, you will be able to:

- distinguish decision agreement from recorded path agreement;
- explain why missing required channels make a replay group ineligible;
- qualify an adapter before spending money on a larger replay;
- run, analyze, and inspect a versioned replay group; and
- separate bounded repeatability evidence from model accuracy.

## The measures

| Measure | Question |
|---|---|
| **DAR** | What fraction of replays match the modal decision? |
| **TARseq** | What fraction match the modal ordered tool-name path? |
| **TARstrong** | What fraction match ordered names, canonical arguments, and result identities? |
| **DAR − TAR gap** | How much more stable is the decision than the recorded path? |

All measures use the same eligible replay denominator. An observed empty path
is data. A missing or malformed required channel is not: the group fails
closed and is excluded from agreement estimates.

## Track A: qualify the packaged integration

From the repository root, create a clean Python 3.10+ environment:

```bash
python -m venv .venv-dfah
source .venv-dfah/bin/activate
python -m pip install -e ".[dev]"
```

Run the no-network conformance check:

```bash
dfah check-agent \
  --agent dfah.demo:toy_agent \
  --episode-timeout-s 5
```

The report checks parse provenance, request/manifest echoing, deterministic
tool outputs, replay-visible ambient-state leakage, resumability, and channel
eligibility.

Now capture three replays and inspect them:

```bash
dfah run \
  --agent dfah.demo:toy_agent \
  --replays 3 \
  --episode-timeout-s 5 \
  --out .dfah/runs/lab-8

dfah analyze .dfah/runs/lab-8 \
  --report .dfah/runs/lab-8/report.html

dfah inspect .dfah/runs/lab-8 --case CASE-001
```

The toy agent should show perfect observed agreement because its policy and
tool are deliberately fixed. That confirms the integration contract. It does
not establish real-world model quality or general determinism.

## Track B: inspect the public research lineage

Use a separate environment for the frozen research pipeline:

```bash
python -m venv .venv-research
source .venv-research/bin/activate
python -m pip install -r requirements.txt

make test-bench
make reproduce-paper
```

The default target regenerates and verifies corrected v2. The raw ledger
contains 8,129 records; its legacy analysis retained 8,127. The corrected
analysis narrows that lineage to 5,501 retained-task episodes and then to
4,157 episodes from configurations with observed tool use. Use
`make reproduce-paper-v1` only for the archived lineage.

Do not compare the package smoke test, historical study, prospective API
diagnostic, and local systems check as one model leaderboard. They use
different tasks, replay counts, and capture contracts.

## What to take away

1. A stable decision can conceal a changing recorded path.
2. Eligibility is part of the estimand; missingness is not agreement.
3. Argument-aware capture can reveal scope changes that name-only paths miss.
4. Replay results describe the tested contract. Accuracy and materiality
   require separate evidence.

## Further reading

- [Interactive explorer](../explorer/index.html)
- [Research papers](../resources/paper.md)
- [Reproducibility notes](https://github.com/ibm-client-engineering/output-drift-financial-llms/blob/main/REPRODUCIBILITY.md)
