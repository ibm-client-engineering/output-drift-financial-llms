---
title: Output Drift in Financial LLMs Workshop
description: Learn how to measure, analyze, and mitigate output drift in financial AI systems
logo: images/ibm-blue-background.png
---

## Output Drift in Financial LLMs Workshop

Welcome to the Output Drift in Financial LLMs Workshop! This hands-on workshop teaches you how to measure and analyze non-determinism in large language model (LLM) outputs for financial applications.

### Why This Matters

Financial institutions deploying AI systems need evidence for:

- **Governance Review**: Reconstructable decisions and execution records
- **Risk Management**: Predictable behavior in production
- **Trust & Reliability**: Stakeholder confidence in AI-driven recommendations

This workshop is based on research showing that LLM outputs can drift even at
temperature=0.0. In the tested tasks, that variation reached 35%, which
complicates replay, change review, and monitoring.

!!! tip "New: interactive results explorer"
    Explore the DFAH-Bench results, play with the replay metrics live in your browser, and drive a real
      local model through the benchmark — no install needed: **[Open the Live Explorer](explorer/index.html)**

!!! tip "New: corrected DFAH-Bench walkthrough"
    Inspect the corrected replay denominator, regenerate v2, and
    compare decision and path agreement: **[Start Lab 8](lab-8/README.md)**.

### What You'll Learn

By the end of this workshop, you will:

* Understand output drift and its implications for financial AI systems
* Set up and run reproducible LLM experiments across multiple providers
* Measure drift using industry-standard metrics (consistency, Jaccard similarity, schema violations)
* Analyze cross-provider reliability patterns
* Implement replay and observability checks for AI deployments

!!! tip
    This workshop is hands-on and collaborative. We encourage you to experiment, ask questions, and share your findings with other participants. The framework is designed to be extensible—feel free to add your own tasks and providers!

## Workshop Structure

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

**Total Duration**: Approximately 3.5-4 hours

## Research Foundation

This workshop is based on three papers from the same research line:

**["Same Decision, Different Path: DFAH-Bench for AI Agents in Finance"](https://arxiv.org/abs/2607.20491)**
[arXiv:2607.20491](https://arxiv.org/abs/2607.20491) |
[DOI](https://doi.org/10.48550/arXiv.2607.20491) |
Paper artifacts reproduce from this repository with `make reproduce-paper`

**"Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents"**
[ICLR 2026 FinAI Workshop](https://sites.google.com/view/iclr2026finai/home) (The 2nd ICLR Workshop on Advances in Financial AI) | [arXiv:2601.15322](https://arxiv.org/abs/2601.15322)

**"LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows"**
Presented at the [AI4F Workshop 2025](https://ai4f-workshop.github.io/) | [arXiv:2511.07585](https://arxiv.org/abs/2511.07585)

**Key Findings (v1, Output Drift):**
- Even at temperature=0.0, frontier models exhibit 5.5-35% output variance
- Several tested 7-20B configurations repeated all sampled outputs at T=0.0
- RAG tasks show the highest drift (56.25% consistency at temperature=0.2)
- Structured output tasks (SQL, summarization) maintain better determinism

**Earlier Replayable Agents study (4,705 runs):**
- Decision determinism and the historical task-label match were not detectably
  correlated (r = -0.11, p = 0.63)
- This descriptive result does not identify model strategy or hidden reasoning
- The portfolio fixture and its dependent task-label matches are not evidence
  in corrected DFAH-Bench v2

**Corrected DFAH-Bench analysis (arXiv v2):**
- The primary slice contains **4,157 episodes from configurations with observed tool use across 719 groups**, eight configurations, and two synthetic tasks
- Among 627 unanimous-decision groups, **122 (19.5%) change tool sequence** and 47 (7.5%) change the tool-name set
- A separate 570-episode API diagnostic finds 94.2–95.1% decision agreement but only 66.9–69.4% exact name-path agreement
- Missing or malformed required channels make a replay group ineligible; they are not treated as agreement

**Community Validation** (Paul Merrison, FINOS):
- Determinism is model-specific, not size-based
- **Gemma2-9B**: 100% deterministic (new Tier 1 candidate)
- **Mistral-7B**: Task-dependent (33% RAG, 100% SQL)
- Architecture and training matter more than parameter count

## Prerequisites

**Required:**
- Python 3.11+
- Basic command line proficiency
- Understanding of APIs and environment variables

**Recommended:**
- Familiarity with LLMs and prompt engineering
- Basic knowledge of financial concepts
- Experience with data analysis (pandas, visualization)

**API Access (at least one):**
- Ollama (free, local)
- IBM watsonx.ai (trial available)
- OpenAI, Anthropic, or other providers

## Target Audience

This workshop is designed for:

- **AI/ML Engineers** building production LLM systems
- **Risk & Compliance Officers** evaluating AI deployments
- **Financial Technologists** integrating AI into workflows
- **Researchers** studying LLM reliability and non-determinism
- **Product Managers** planning AI-powered financial products

## Getting Help

If you encounter issues or have questions:

1. Check the [Troubleshooting Guide](resources/troubleshooting.md)
2. Review the [API Reference](resources/api.md)
3. Ask workshop facilitators or teaching assistants
4. Open an [Issue](https://github.com/ibm-client-engineering/output-drift-financial-llms/issues/new) on GitHub
5. Submit a [Pull Request](https://github.com/ibm-client-engineering/output-drift-financial-llms/pulls) with improvements

## Repository Structure

```
output-drift-financial-llms/
├── run_evaluation.py       # Main experiment orchestrator (v1 Output Drift)
├── run_dfah_demo.py        # DFAH demo (no LLM needed)
├── Makefile                # make reproduce-paper / make test-bench
├── REPRODUCIBILITY.md      # Exact environment, commands, disclosed caveats
├── COMMUNITY_FINDINGS.md   # Independent validation results
├── docs/                   # Workshop documentation (labs 0-8)
├── bench/                  # DFAH-Bench library (v3)
│   ├── metrics/            # DAR/TAR, ECD, DCB, SCDR implementations
│   ├── spec/               # Replay episode schema + task ontologies
│   ├── provenance/         # Hash-chained, Ed25519-signed audit bundles
│   └── stats/              # Bootstrap CIs, significance tests
├── src/dfah/               # Prospective pip-installable replay package
├── harness/                # v1 framework core
│   ├── deterministic_retriever.py
│   ├── task_definitions.py
│   └── cross_provider_validation.py
├── providers/              # LLM providers (watsonx, anthropic, gemini)
├── econometrics/           # Replayable Agents (v2) - benchmarks & metrics
│   ├── benchmarks/         # 3 financial agent benchmarks (50 cases each)
│   │   └── results/run_logs/  # Raw replay corpus: 8,129 episodes
│   └── agentic/            # Trajectory determinism & faithfulness metrics
├── scripts/                # Replay analysis + reproduce_paper.py
├── tests/                  # Offline research and package tests
├── results/                # Reference CSVs behind every paper number
├── data/                   # Test datasets & generators
├── examples/               # Audit trails + domain-extension example
└── requirements.txt        # Python dependencies
```

## Reproducibility & Citations

The v1 drift experiments are pinned at **release v0.1.0 (commit c19dac5)**:

```bash
git clone https://github.com/ibm-client-engineering/output-drift-financial-llms
git checkout v0.1.0
```

The sanitized DFAH-Bench replay fixture is checked in. The default target
regenerates the corrected v2 retrospective slice and verifies the
aggregate-only extensions and manifest:

```bash
make reproduce-paper
make reproduce-paper-v1  # archived v1 lineage only
```

If you use this framework in your research, please cite:

```bibtex
@article{khatchadourian2026replayable,
  title={Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents},
  author={Khatchadourian, Raffi},
  journal={arXiv preprint arXiv:2601.15322},
  year={2026},
  eprint={2601.15322},
  archivePrefix={arXiv},
  doi={10.48550/arXiv.2601.15322}
}

@article{khatchadourian2025output,
  title={LLM Output Drift: Cross-Provider Validation \& Mitigation for Financial Workflows},
  author={Khatchadourian, Raffi and Franco, Rolando},
  journal={arXiv preprint arXiv:2511.07585},
  year={2025},
  eprint={2511.07585},
  archivePrefix={arXiv},
  doi={10.48550/arXiv.2511.07585}
}
```

```bibtex
@article{khatchadourian2026dfahbench,
  title={Same Decision, Different Path: DFAH-Bench for AI Agents in Finance},
  author={Khatchadourian, Raffi},
  journal={arXiv preprint arXiv:2607.20491},
  year={2026},
  eprint={2607.20491},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  doi={10.48550/arXiv.2607.20491},
  url={https://arxiv.org/abs/2607.20491}
}
```

**DFAH-Bench**: [arXiv:2607.20491](https://arxiv.org/abs/2607.20491) |
**Replayable Agents**: [arXiv:2601.15322](https://arxiv.org/abs/2601.15322) |
**Output Drift**: [arXiv:2511.07585](https://arxiv.org/abs/2511.07585)

## License

This project is licensed under the MIT License. See the
[repository license](https://github.com/ibm-client-engineering/output-drift-financial-llms/blob/main/LICENSE)
for details.

## Contributors & Acknowledgments

This workshop and framework were developed by Raffi Khatchadourian and Rolando Franco in IBM Financial Services in collaboration with researchers focused on responsible AI deployment in regulated industries.

Special thanks to the open-source community and the contributors who helped build and test this framework.

---

!!! success "Ready to Begin?"
    Start with [Lab 0: Workshop Pre-work](pre-work/README.md) to set up your environment!
