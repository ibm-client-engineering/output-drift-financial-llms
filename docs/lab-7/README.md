# Lab 7: Replayable Financial Agents (ICLR 2026)

## Overview

This lab introduces the **Replayable Financial Agents** research track, extending the Output Drift framework from single-turn tasks (Labs 1-6) to multi-step, tool-using LLM agents. This work is based on the [ICLR 2026 FinAI Workshop](https://sites.google.com/view/iclr2026finai/home) paper.

!!! warning "Historical study boundary"
    This lab preserves the earlier three-task study. Its portfolio fixture and
    dependent task-label-match results are not part of the corrected
    DFAH-Bench v2 analysis. Use Lab 8 for the corrected evidence.

**Paper**: [arXiv:2601.15322](https://arxiv.org/abs/2601.15322)

**Duration**: ~30 minutes (benchmark simulation) or ~2+ hours (full agentic runs with LLMs)

## Learning Objectives

By the end of this lab, you will:

- Understand how determinism extends from single-turn outputs to multi-step agent trajectories
- Run the 3 financial agent benchmarks (Compliance Triage, Portfolio Constraint, DataOps Exception)
- Measure trajectory determinism, decision determinism, and historical
  task-label match
- Distinguish final-decision agreement from recorded tool-path agreement

## Key Concepts

### From Output Drift to Agent Determinism

Labs 1-6 measured whether the same prompt produces the same output. Agent determinism asks a harder question: **does an agent take the same actions and reach the same conclusion when given the same task?**

| Metric | Definition | What It Captures |
|--------|-----------|------------------|
| **Signature Determinism** | Identical tool calls with identical arguments | Exact trajectory reproducibility |
| **Decision Determinism** | Same final action (e.g., escalate/dismiss) | Outcome consistency |
| **Historical task-label match** | Decision matches the fixture label | Earlier-study diagnostic |

### Separate repeatability from task-label evidence

The earlier study reported across 4,705 agentic runs:

> Decision determinism and task accuracy are **not detectably correlated** (r = -0.11, p = 0.63).

The null result does not show that the two quantities are equivalent or that
either validates deployment. The study used the following descriptive
groupings; they do not establish hidden reasoning:

| Observed grouping | Example Models | Determinism | Historical task-label match | Recorded behavior |
|---------|---------------|-------------|----------|----------|
| **High repeatability** | Qwen 2.5 7B, Granite 3.3 | 91-98% | 33-42% | Concentrated decisions in these tasks |
| **Moderate repeatability** | Claude Sonnet 4, Gemini Flash | 77-86% | 33-53% | Some recorded path variation |
| **Variable paths** | Claude Opus 4, Gemini 2.5 Pro | 59-71% | 40-69% | Highest observed signature variability in these tasks |

## Prerequisites

- Completed Labs 0-3 (environment setup, basic experiments)
- Python environment with `requirements.txt` installed

## Step 1: Understand the Benchmarks

The framework includes 3 financial agent benchmarks, each with 50 test cases:

### Compliance Triage

A compliance agent receives a suspicious transaction alert and must decide: **escalate**, **dismiss**, or **investigate**.

Tools available: `check_sanctions_list`, `get_customer_profile`, `search_precedents`, `calculate_risk_score`

### Portfolio Constraint

A portfolio agent evaluates proposed trades against position limits, sector caps, and liquidity requirements. Decision: **approve**, **reject**, or **modify**.

Tools available: `get_current_holdings`, `get_market_data`, `check_position_limit`, `calculate_sector_exposure`, `get_regulatory_constraints`

### DataOps Exception

A data operations agent resolves data quality exceptions in a financial pipeline. Decision: **auto_fix**, **escalate**, or **quarantine**.

Tools available: `get_exception_details`, `query_reference_data`, `get_historical_fixes`, `validate_fix`, `apply_fix`, `escalate_to_human`

## Step 2: Run the Deterministic Benchmark Simulation

The benchmark suite includes a **deterministic simulation mode** that runs without an LLM. This tests the benchmark infrastructure and demonstrates the metrics computation:

```bash
# Run all 3 benchmarks (no LLM needed, completes in seconds)
python econometrics/benchmarks/run_all.py
```

**Expected output:**

```
============================================================
V3 FINANCIAL AGENT BENCHMARK SUITE
============================================================
Configuration: 5 runs/test, max 50 tests

Running benchmark: compliance_triage
Loaded 50 test cases

COMPLIANCE_TRIAGE RESULTS:
  Tests: 50
  Runs per test: 5
  Action Determinism: 100.0%
  Signature Determinism: 100.0%
  Decision Determinism: 100.0%
  Accuracy: XX.X%
```

!!! info "Why 100% Determinism?"
    The simulation mode uses deterministic agent logic (no LLM), so it always produces identical results. This validates the benchmark infrastructure. Real LLM-driven runs will show the behavioral differences described in the paper.

### Run a single benchmark:

```bash
python econometrics/benchmarks/run_all.py --task compliance_triage --n-runs 8
```

## Step 3: Run Agentic Benchmarks with an LLM (Optional)

!!! warning "API Costs"
    Running the full benchmark suite (50 cases x 8 runs x 3 benchmarks = 1,200 runs per model) requires significant API calls. Start small with `--n-cases 5` to test, then scale up. Our full v2 experiments cost ~$66 across all providers.

To run benchmarks with actual LLM tool-calling (requires Ollama):

```bash
# Small test run (5 cases, fast)
python econometrics/benchmarks/run_agentic_benchmark.py \
  --model qwen2.5:7b-instruct \
  --n-cases 5 \
  --n-runs 4

# Larger run (50 cases, matches paper methodology)
python econometrics/benchmarks/run_agentic_benchmark.py \
  --model qwen2.5:7b-instruct \
  --n-cases 50 \
  --n-runs 8
```

Results are saved to `econometrics/benchmarks/results/`.

## Step 4: Interpret Results

### Reading the Output

Each benchmark run produces:

- **Action Determinism**: Do all runs call the same set of tools?
- **Signature Determinism**: Do all runs call tools with the same arguments in the same order?
- **Decision Determinism**: Do all runs reach the same final decision?
- **Historical task-label match**: How often does the decision match the
  fixture label? This is retained only for the earlier study.

### What to Look For

**High repeatability + lower task-label match**:
The tested configuration often repeats the same decision, but that decision is
not necessarily correct.

**Moderate repeatability + higher task-label match**:
The recorded tool path and decision vary more often. The logs show the
variation, not the model's hidden reason for it.

**Lower repeatability + variable task-label match**:
The recorded path changes across runs. Inspect which calls or arguments
changed before deciding whether the variation matters.

### The "Same Conclusion, Different Recorded Path" Pattern

Across the API configurations, **decision determinism exceeds signature
determinism**. For example, Claude Sonnet 4 shows 84% decision determinism but
43% signature determinism across the earlier benchmarks. The final decision
can therefore stay fixed while observable tool calls change.

## Step 5: Explore the Metrics Modules

The determinism metrics are implemented in reusable modules:

```python
# Trajectory determinism computation
from econometrics.agentic.metrics.trajectory_determinism import (
    compute_trajectory_determinism  # if available
)

# Faithfulness metrics
from econometrics.agentic.metrics.faithfulness import (
    compute_faithfulness  # if available
)
```

See `econometrics/agentic/metrics/` for the full implementation.

## Key Takeaways

1. **Repeatability is not correctness**: each requires separate evidence
2. **Study versions matter**: the corrected DFAH-Bench result excludes the
   portfolio fixture and its dependent task-label matches
3. **Task and contract matter**: report them alongside every replay measure
4. **Recorded paths can differ**: Decision determinism can exceed signature determinism
5. **Start small**: Use `--n-cases 5` to validate your setup before scaling to full 50-case runs

## Further Reading

- **Full paper**: [arXiv:2601.15322](https://arxiv.org/abs/2601.15322) (Replayable Financial Agents)
- **Benchmark details**: [`econometrics/benchmarks/README.md`](https://github.com/ibm-client-engineering/output-drift-financial-llms/blob/main/econometrics/benchmarks/README.md)
- **v1 Output Drift paper**: [arXiv:2511.07585](https://arxiv.org/abs/2511.07585)

---

!!! success "Lab 7 Complete!"
    You now understand how determinism extends from single-turn outputs to multi-step agent trajectories, and how to run and interpret the financial agent benchmarks from the Replayable Financial Agents paper.
