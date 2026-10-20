---
title: DFAH — Replay, Evidence, and Outcomes
description: Explore the DFAH studies and learn to measure repeated decisions, execution records, and completed work.
---

# What does a successful agent run tell us?

An agent can repeat its answer while changing the work behind it. Its records
can also agree because they leave out the same calls. Understanding outcomes
means examining what ran, whether the record preserves it, and whether the
available evidence supports the action.

DFAH connects these questions through replay experiments, capture diagnostics,
and banking studies. The labs turn the distinctions into examples you can run.

!!! tip "Explore the story"
    [Open the interactive explorer](explorer/index.html) to compare recorded
    paths and see how the capture and banking findings fit together.

## Two 2026 workshop papers

| Accepted poster | Main question | Start here |
|---|---|---|
| **AI4F at ACM ICAIF** — [Evidence Before Execution in Financial Agent Gates](https://openreview.net/forum?id=VWQqIgrKw4) | Does the evidence justify an action, and what happens to completion and cost? | [Banking study](dfah/v3-study.md), [Lab 10](lab-10/README.md) |
| **SLM-Agents at NeurIPS** — [When Replay Agreement Misleads](https://openreview.net/forum?id=njEKVxonV2) | Can averages hide path changes and repeated records hide omissions? | [Study and supplement](dfah/workshop-studies.md), [Lab 11](lab-11/README.md) |

These papers develop different parts of DFAH and share some historical
observations. The [research guide](resources/paper.md) connects them with the
preprint and earlier work.

## Choose a starting point

- **Try a 15-minute example without installing packages:** [Lab 11](lab-11/README.md)
  uses Python's standard library to show capture loss.
- **Measure your agent's replays:** start with the [package quickstart](dfah/quickstart.md)
  and [Lab 8](lab-8/README.md).
- **Follow the evidence through a workflow:** [Lab 9](lab-9/README.md) inspects path
  changes; [Lab 10](lab-10/README.md) separates proposed and completed calls.
- **Study output drift across providers:** Labs 1–6 use the earlier output-drift
  framework; [Lab 0](pre-work/README.md) explains setup and model access.

## What the findings show

The prospective replay diagnostic retained **570 episodes**. Decisions agreed
at **94–95%**, while paths including arguments and results agreed at **45–52%**.
One condition missed the planned minimum by one group, so the comparison is
diagnostic.

In **800 historical Qwen episodes**, decision and path agreement both averaged
**99.625%**, yet opposing case-level gaps hid a path change. In the separate
**48-execution capture study**, deleting the same calls from repeated records
left agreement at **100%**. Compare the per-case evidence and the execution
reference to see what those averages leave out.

The banking study scheduled **1,080 episodes**, with **1,033 known outcomes**.
The typed-choice bundle cost less on complete task pairs but had fewer primary
successes than the generative bundle under every assignment of unknown
outcomes. Input limits and recovery behavior are part of those setups;
[the banking guide](dfah/v3-study.md) keeps the comparison populations explicit.

## Labs

| Lab  | Description  | Duration |
| :--- | :--- | :--- |
| [Lab 0: Workshop Pre-work](pre-work/README.md) | Install prerequisites and set up your environment | 15 min |
| [Lab 1: Understanding Output Drift](lab-1/README.md) | Learn the theory and see real examples of drift | 20 min |
| [Lab 2: Setting Up Your Environment](lab-2/README.md) | Configure API keys and run environment tests | 15 min |
| [Lab 3: Running Your First Experiment](lab-3/README.md) | Execute experiments and understand the framework | 30 min |
| [Lab 4: Analyzing Drift Metrics](lab-4/README.md) | Interpret results and generate visualizations | 25 min |
| [Lab 5: Cross-Provider Testing](lab-5/README.md) | Compare reliability across different AI providers | 30 min |
| [Lab 6: Extending the Framework](lab-6/README.md) | Add custom tasks and integrate with your workflows | 30 min |
| [Lab 7: Replayable Financial Agents](lab-7/README.md) | Run agent benchmarks from the ICLR 2026 paper | 30 min |
| [Lab 8: DFAH-Bench — Replay Measurement](lab-8/README.md) | Test the package, then reproduce the paper from the checked-in replay corpus | 30 min |
| [Lab 9: Replay, Review, and Retest](lab-9/README.md) | Evaluate a correction with a fixed policy and export verified results | 25 min |
| [Lab 10: Evidence Before Execution](lab-10/README.md) | Inspect proposals, completed calls, and missing records | 15 min |
| [Lab 11: Capture Loss and Agreement](lab-11/README.md) | See how repeated omissions preserve agreement, then compare against execution evidence | 15 min |

Allow roughly five hours for the whole sequence, or take a single lab.
The package track in Lab 8 and Labs 9–11 do not need a model service or API key.
Provider experiments in earlier labs require the corresponding model access.

## Recompute the research

The public historical fixture supports:

```bash
make reproduce-paper
make verify-v2-manifest
```

Use the repository's research dependencies for those commands. The separate
[SLM supplement](https://openreview.net/attachment?id=njEKVxonV2&name=supplementary_material)
provides standard-library verifiers for its historical summaries and capture
projections. [Read the reproduction guide](dfah/workshop-studies.md) to select
the right material. The hosted banking inputs are not released here.

## Earlier studies

- [DFAH-Bench preprint](https://arxiv.org/abs/2607.20491): replay, capture, and authorization evidence.
- [Replayable Financial Agents](https://arxiv.org/abs/2601.15322): the earlier ICLR 2026 FinAI study, used in Lab 7.
- [LLM Output Drift](https://arxiv.org/abs/2511.07585): the AI4F 2025 study behind Labs 1–6.

## Get help

See [troubleshooting](resources/troubleshooting.md), the
[API reference](resources/api.md), or the
[community findings](resources/community-findings.md). Report a problem through
[GitHub issues](https://github.com/ibm-client-engineering/output-drift-financial-llms/issues).

The software is distributed under the [repository license](https://github.com/ibm-client-engineering/output-drift-financial-llms/blob/main/LICENSE).
External research materials retain their own terms.
