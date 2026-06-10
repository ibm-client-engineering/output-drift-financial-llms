# Lab 8: DFAH-Bench — Replay Determinism & Audit Bundles

## Overview

This lab introduces **DFAH-Bench**, the replay benchmark behind our newest
paper, *DFAH-Bench: Benchmarking Observable Agent Instability in Financial
Decision-Making* (arXiv preprint). It is the deepest layer of the framework:
where Lab 7 measured whether an agent is deterministic, DFAH-Bench measures
**how** it diverges — by decision, trajectory, or evidence — and packages the
proof in verifiable audit bundles.

**The headline finding**: outcome-only evaluation reports stable agents whose
reasoning is not stable. Across 8,127 replay episodes, **21.8% of
decision-stable case groups hide trajectory divergence** that final-answer
metrics cannot see.

**Duration**: ~25 minutes. **No LLM, no API keys, no network needed** — the
full 8,129-episode replay corpus is checked into this repository, and every
number in the paper regenerates from it on your laptop.

## Learning Objectives

By the end of this lab, you will:

- Distinguish the three observable divergence channels: decision, trajectory, evidence
- Regenerate every number in the DFAH-Bench paper with one command
- Read a channel-availability matrix and know why it must accompany results
- Extend the benchmark to a brand-new domain with zero metric-code changes

## Key Concepts

### The Metrics

| Metric | Definition | Failure mode it exposes |
|--------|-----------|------------------------|
| **DAR** | Decision Agreement Rate — fraction of N replays agreeing with the modal decision | Outcome instability |
| **TAR** | Trajectory Agreement Rate — fraction matching the modal tool-call sequence exactly | Process instability |
| **DAR − TAR gap** | The central diagnostic | "Same conclusion, different path" |
| **ECD** | Evidence-Contact Divergence — mean pairwise Jaccard distance over evidence sets | "Same conclusion, different evidence" |
| **DCB** | Decision Concentration Bias — 1 − H(p)/log K | Pattern matching (always picking the same label) |

### Why the gap matters

A model can score DAR = 0.947 (outcomes look rock-solid) while TAR = 0.767 —
meaning nearly a quarter of its replays took a *different tool path* to that
same answer. In a demo that's invisible. In an audit, the path is the product.

## Step 1: Run the test suite

```bash
make test-bench
# 143 tests: metric math, schema round-trips, hash-chain tamper vectors,
# Ed25519 certificates, pipeline conventions — all offline
```

## Step 2: Reproduce the paper

```bash
make reproduce-paper-fast    # ~2 min: skips the B=10,000 bootstrap
make reproduce-paper         # full verification incl. bootstrap CIs
```

Watch for the five stages: episode accounting (8,129 raw → 8,127 analyzed),
pipeline regeneration, reference diff, paper-claim assertions, and the
C(8,3)=56 subsampling robustness check. Every `PASS` line is a paper number
regenerating from raw logs. The command **fails loudly** if anything drifts.

## Step 3: Read the kill criterion

```bash
cat results/dfah_kill_criterion.csv
cat results/dfah_kill_criterion_by_model.csv
```

Among 912 case groups with high decision agreement (DAR ≥ 0.9): 21.8% show
trajectory divergence (TAR < 0.9), 19.4% show strong divergence (TAR < 0.7).
Per model: 55.6% of Claude Sonnet's and 56.6% of Gemini 2.5 Pro's eligible
cases diverge. These are the numbers in §4.4 of the paper — now on your screen.

## Step 4: Extend to a new domain

```bash
python examples/domain_extension_medical.py
```

This registers a **medical triage** ontology (escalate / treat / refer) with
two mock tools and computes every DFAH metric with the unmodified `bench/`
library. CASE-B in the output is the paper's central failure mode reproduced
in a domain the benchmark has never seen: DAR = 1.000, TAR = 0.333.

To add your own domain: define a `DecisionOntology` + `TaskSpec`, call
`register_task(spec)`, and feed it replay episodes. No metric code changes.

## Step 5: Verify an audit bundle

```bash
python -m bench.provenance.verify --help
```

Audit bundles are SHA-256 hash-chained and Ed25519-signed (the provenance
layer depends only on the standard library + `cryptography`). The test suite
includes six tamper vectors — payload modification, deletion, reordering,
backward timestamps, genesis tampering, duplicate sequence numbers — all
detected.

## What to take away

1. **Decision stability ≠ trajectory stability.** Measure both, per channel.
2. **Channel availability is part of the result.** Metrics only run on
   channels actually present — and the matrix saying which is published
   alongside every table.
3. **Reproducibility is a command, not a promise.** If `make reproduce-paper`
   passes, the paper's numbers are your numbers.

## Bonus: the Live Explorer

Prefer to see it before you clone it? The **[interactive explorer](../explorer/index.html)** renders the
DAR-TAR scatter and kill criterion from these same results, lets you feel the gap with a seeded replay
simulator, and can even drive your local Ollama model through the compliance case live in the browser.

## Further Reading

- [`REPRODUCIBILITY.md`](https://github.com/ibm-client-engineering/output-drift-financial-llms/blob/main/REPRODUCIBILITY.md) — exact environment, commands, disclosed caveats
- [`DFAH.md`](https://github.com/ibm-client-engineering/output-drift-financial-llms/blob/main/DFAH.md) — harness documentation
- [Research Papers](../resources/paper.md) — all three papers in this research line
