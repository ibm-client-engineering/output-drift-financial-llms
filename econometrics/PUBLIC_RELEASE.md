# V3 Public Release Package

**Version**: 1.0.0
**Date**: December 19, 2025
**Target**: ICLR 2026 FinAI Workshop Supplementary Materials

---

## Package Contents

### Benchmark Tasks (50 test cases each)

```
econometrics/benchmarks/
├── compliance_triage/
│   ├── task.py                      # Task definition and tools
│   └── data/alerts.json             # 50 compliance alerts
├── portfolio_constraint/
│   ├── task.py                      # Task definition and tools
│   └── data/trades.json             # 50 proposed trades
├── dataops_exception/
│   ├── task.py                      # Task definition and tools
│   └── data/exceptions.json         # 50 data exceptions
├── run_all.py                       # Unified benchmark runner
└── tier_experiment.py               # V2-V3 connection experiment
```

### Metrics Modules

```
econometrics/
├── drift_variance_estimator.py      # Label stability measurement
├── semantic_divergence_econometric.py  # Semantic vs lexical drift
├── validation_debiasing.py          # Ludwig et al. debiasing
└── leakage_detection.py             # Training data leakage check
```

### Agentic Evaluation

```
econometrics/agentic/
├── metrics/
│   ├── trajectory_determinism.py    # Tool sequence analysis
│   └── faithfulness.py              # Evidence grounding
└── harness/
    └── stress_test_runner.py        # Perturbation testing
```

### Paper Materials

```
econometrics/paper/
├── latex/replayable_agents.tex      # ICLR 2026 submission (8 pages)
├── ICLR_FINAI_2026_OUTLINE.md       # Paper structure
└── references.bib                    # Bibliography
```

---

## Quick Start

### 1. Run Benchmark Suite

```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Run all benchmarks (quick test)
python econometrics/benchmarks/run_all.py --max-tests 10 --n-runs 3

# Full benchmark run
python econometrics/benchmarks/run_all.py --max-tests 50 --n-runs 10
```

### 2. Run Tier Experiment (V2-V3 Connection)

```bash
# Connect to V2 data and project benchmark performance
python econometrics/benchmarks/tier_experiment.py
```

### 3. Validate Modules

```bash
# Econometric track
python econometrics/drift_variance_estimator.py
python econometrics/semantic_divergence_econometric.py
python econometrics/validation_debiasing.py

# Agentic track
python -c "from econometrics.agentic.metrics.trajectory_determinism import example_compliance_agent_determinism; example_compliance_agent_determinism()"
python -c "from econometrics.agentic.metrics.faithfulness import example_portfolio_agent_faithfulness; example_portfolio_agent_faithfulness()"
```

### 4. Compile Paper

```bash
cd econometrics/paper/latex
pdflatex replayable_agents.tex
```

---

## Key Findings Summary

| Tier | Determinism | Faithfulness | Validation Scaling |
|------|-------------|--------------|-------------------|
| Tier 1 (7-20B) | **100%** | **100%** | 1.0x |
| Frontier | 88.5% | 100% | 1.34x |
| Tier 2 (40-70B) | 73.4% | 75% | 1.8x |
| Tier 3 (120B+) | 9.7% | 71.9% | **3.7x** |

**Task-Structure Effect**:
- Portfolio Constraint (structured): Highest determinism
- DataOps Exception (semi-structured): Medium determinism
- Compliance Triage (open-ended): Lowest determinism

**Critical Finding**: Positive correlation (r=0.45) between determinism and faithfulness - no trade-off!

---

## Citation

If you use this benchmark or harness in your research, please cite:

```bibtex
@inproceedings{replayable2026,
  title={Replayable Financial Agents: A Determinism-Faithfulness Assurance
         Harness for Tool-Using LLM Agents},
  author={Anonymous},
  booktitle={ICLR 2026 Workshop on AI for Financial Services},
  year={2026}
}
```

---

## License

Apache 2.0 - See LICENSE file

---

## Contact

- GitHub Issues: https://github.com/ibm-client-engineering/output-drift-financial-llms/issues
- Workshop: ICLR 2026 FinAI Workshop, Rio de Janeiro, April 27 - May 1, 2026
