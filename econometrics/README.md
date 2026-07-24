# Replayable Agents (v2): Econometric & Agentic Analysis of LLM Drift

> **v2 Extension** of the [LLM Output Drift framework](https://arxiv.org/abs/2511.07585) (v1, AI4F Workshop 2025). This module adds econometric methods for LLMs as measurement instruments and replayability/auditability frameworks for tool-using agents.

---

## Version History

| Version | Focus | Status |
|---------|-------|--------|
| **v1** | Output drift measurement (single-turn tasks) | Published (AI4F Workshop 2025) |
| **v2** | Agent trajectory determinism & faithfulness | Current |
| **v3** | Panel econometrics for drift-contaminated data | Planned |

---

## Overview

This module extends the empirical findings from the LLM Output Drift paper into two complementary research directions:

1. **Econometrics Track**: Rigorous framework for using LLMs as measurement instruments in social science research (extends Ludwig et al. 2024)

2. **Agentic Track**: Replayability and auditability for tool-using LLM agents in financial workflows

## Research Motivation

Our prior work established empirically that:
- **7-20B parameter models achieve 100% determinism** at T=0.0, while 120B+ MoE architectures show only 12.5% consistency
- **Task structure matters**: Schema-constrained outputs (SQL, JSON) achieve near-perfect determinism regardless of model size
- **Faithfulness and determinism are orthogonal**: "Varying-but-correct" outputs are preferable to "consistent-but-wrong" for downstream applications

**This project asks**: What are the econometric implications of these findings for researchers using LLMs to generate labels, scores, and classifications for downstream analysis?

## Core Research Questions

### 1. Measurement Error Decomposition
Standard econometric measurement error models assume:
```
Y_llm = Y_true + ε_model  (deterministic conditional on input)
```

We propose a three-component decomposition:
```
Y_llm = Y_true + ε_capability + ε_drift
```

Where:
- `ε_capability`: Systematic model error (capability limitations)
- `ε_drift`: Run-to-run variance for identical inputs (our contribution)

**Key Insight**: For Tier 1 models (100% determinism), `ε_drift = 0`. For Tier 3 models (12.5% determinism), `ε_drift` dominates.

### 2. Validation Subsample Implications
Following Ludwig, Mullainathan & Rambachan (2024), validation subsamples are essential for debiasing LLM-generated labels. Our findings imply:

| Model Tier | Determinism | Validation Scaling Factor |
|------------|-------------|---------------------------|
| Tier 1 (7-20B) | 100% | 1.0x (baseline) |
| Tier 2 (20-70B) | 50-87% | 1.2-1.5x |
| Tier 3 (120B+) | 12.5% | 2.5-3.0x |

High-drift models require larger validation samples to achieve equivalent debiasing precision.

### 3. Semantic vs. Lexical Drift
Not all drift is created equal:
- **Semantic drift**: Changes in meaning (affects downstream estimates)
- **Lexical drift**: Paraphrasing variance (no effect on well-designed measures)

We develop lightweight TF-IDF-based semantic divergence metrics (no transformers required) to distinguish these cases.

### 4. Leakage Detection for Prediction Tasks
For prediction tasks (vs. estimation), training data leakage invalidates out-of-sample performance. We implement:
- Temporal leakage detection (dates after model cutoff)
- N-gram fingerprinting for exact matches
- Fuzzy matching for near-duplicates

## Module Overview

### Core Modules

| Module | Description | Key Contribution |
|--------|-------------|------------------|
| `drift_variance_estimator.py` | Multi-run drift analysis | Optimal k runs, majority voting |
| `semantic_divergence_econometric.py` | Semantic equivalence classes | Validation cost reduction |
| `validation_debiasing.py` | Drift-augmented debiasing | Extends Ludwig et al. (2024) |
| `leakage_detection.py` | Training data leakage | Temporal + n-gram detection |

## Relationship to Ludwig et al. (2024)

This work builds directly on the foundational framework in:

> Ludwig, J., Mullainathan, S., & Rambachan, A. (2024). Large Language Models: An Applied Econometric Framework. arXiv:2412.07031

**Key extensions**:
1. We decompose their `ε_model` into `ε_capability + ε_drift`
2. We provide empirical calibration of drift variance by model tier
3. We introduce semantic equivalence classes to reduce validation costs
4. We link their framework to specific model selection recommendations

## Quick Start

```python
from econometrics.drift_variance_estimator import run_multi_run_analysis
from econometrics.validation_debiasing import debiased_ols_regression, ValidationSample

# 1. Estimate drift from multiple LLM runs
analysis = run_multi_run_analysis(
    sample_ids=sample_ids,
    runs_matrix=llm_runs,  # shape (n_samples, k_runs)
    categorical=True,
    drift_threshold=0.10
)
print(analysis.summary())

# 2. Create validation sample
validation = ValidationSample(
    texts=validation_texts,
    y_true=human_labels,
    y_llm=llm_labels,
    y_llm_runs=multi_run_labels
)

# 3. Debiased regression
results = debiased_ols_regression(
    y_llm_full=all_llm_labels,
    X_full=returns,
    validation=validation,
    include_drift=True
)
print(f"Naive beta: {results.beta_naive[1]:.3f}")
print(f"Drift-corrected beta: {results.beta_drift_debiased[1]:.3f}")
```

## Data Requirements

This framework requires:
1. **Multi-run LLM outputs**: Run each sample k times (k=3-10) to estimate drift
2. **Validation subsample**: Human-labeled subset (10-20% of data)
3. **Outcome variable**: For downstream regression (e.g., returns, spreads)

## Citation

If building on the drift findings:
```bibtex
@misc{khatchadourian2025llm,
  title={LLM Output Drift: Cross-Provider Validation \& Mitigation for Financial Workflows},
  author={Khatchadourian, Raffi and Franco, Rolando},
  year={2025},
  howpublished={AI4F Workshop 2025, Singapore},
  eprint={2511.07585},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2511.07585}
}
```

If using the econometric framework:
```bibtex
@article{ludwig2024llm,
  title={Large Language Models: An Applied Econometric Framework},
  author={Ludwig, Jens and Mullainathan, Sendhil and Rambachan, Ashesh},
  journal={arXiv preprint arXiv:2412.07031},
  year={2024}
}
```

---

## Agentic Track: Replayable Financial Agents

*Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents*
**Accepted**: [ICLR 2026 FinAI Workshop](https://sites.google.com/view/iclr2026finai/home) | [arXiv:2601.15322](https://arxiv.org/abs/2601.15322)

### Thesis

Agentic AI in finance becomes auditable and safe when we can replay an agent's tool-using trajectory and quantify the determinism-faithfulness frontier under shocks, model/provider changes, and data-quality perturbations.

### Three Contributions

1. **Agent Replayability Metric Suite** (Novel)
   - Trajectory Determinism: identical tool-call sequences
   - Decision Determinism: identical final actions
   - Evidence-Conditioned Faithfulness: grounded in evidence + constraints

2. **Drift Stress-Test Harness** (Novel)
   - Model/provider updates
   - Data shifts (stale filings, schema changes)
   - Data-quality faults (missing fields, inconsistent IDs)
   - Market shocks (rate spikes, volatility jumps)

3. **Controls That Work** (Engineering + Science)
   - Unconstrained ReAct (baseline)
   - Schema-First / Typed-Tool agents
   - Policy-Gated agents

### Key References

- Xiang et al. (2024): Scientific Discovery - "LLMs optimize for plausibility, not truth"
- OpenAI (2024): CoT Monitorability - Reasoning may not reflect decision process
- Ludwig et al. (2024): Econometric Framework for LLMs

---

## Module Structure

```
econometrics/
├── README.md                           # This file
├── __init__.py                         # Package exports
│
├── # Econometric Track
├── drift_variance_estimator.py         # Multi-run drift analysis
├── semantic_divergence_econometric.py  # Semantic equivalence classes
├── validation_debiasing.py             # Ludwig et al. extension
├── leakage_detection.py                # Training data leakage
│
├── # Agentic Track
├── agentic/
│   ├── metrics/
│   │   ├── trajectory_determinism.py   # Trajectory/decision determinism
│   │   └── faithfulness.py             # Evidence-conditioned faithfulness
│   └── harness/
│       └── stress_test_runner.py       # Drift stress-test harness
│
└── benchmarks/                         # Financial agent benchmarks
    ├── run_all.py                      # Run all 3 benchmarks
    ├── run_agentic_benchmark.py        # Agentic (tool-calling) benchmark runner
    ├── compliance_triage/              # Compliance triage task + 50 test cases
    ├── portfolio_constraint/           # Portfolio constraint task + 50 test cases
    └── dataops_exception/              # DataOps exception task + 50 test cases
```

---

## License

MIT License - See parent repository.
