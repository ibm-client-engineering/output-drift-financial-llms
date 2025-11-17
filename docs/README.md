---
title: Output Drift in Financial LLMs Workshop
description: Learn how to measure, analyze, and mitigate output drift in financial AI systems
logo: images/ibm-blue-background.png
---

## Output Drift in Financial LLMs Workshop

Welcome to the Output Drift in Financial LLMs Workshop! This hands-on workshop teaches you how to measure and analyze non-determinism in large language model (LLM) outputs for financial applications.

### Why This Matters

Financial institutions deploying AI systems must ensure:
- **Regulatory Compliance**: Consistent, auditable AI decisions
- **Risk Management**: Predictable behavior in production
- **Trust & Reliability**: Stakeholder confidence in AI-driven recommendations

This workshop is based on peer-reviewed research demonstrating that even at temperature=0.0, LLMs exhibit output drift—up to 35% variance in some tasks—threatening compliance workflows.

### What You'll Learn

By the end of this workshop, you will:

* Understand output drift and its implications for financial AI systems
* Set up and run reproducible LLM experiments across multiple providers
* Measure drift using industry-standard metrics (consistency, Jaccard similarity, schema violations)
* Analyze cross-provider reliability patterns
* Implement best practices for deterministic AI deployments

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

**Total Duration**: Approximately 2.5-3 hours

## Research Foundation

This workshop is based on the peer-reviewed paper:

**"LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows"**

📄 [Read the full paper on arXiv](https://arxiv.org/abs/2511.07585)

**Key Findings:**
- Even at temperature=0.0, frontier models exhibit 5.5-35% output variance
- 7-8B models (Granite-3-8B, Qwen2.5-7B) achieve 100% determinism at T=0.0
- RAG tasks show the highest drift (56.25% consistency at temperature=0.2)
- Structured output tasks (SQL, summarization) maintain better determinism
- Cross-provider experiments reveal significant reliability gaps

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
├── run_evaluation.py       # Main experiment orchestrator
├── make_tables.py          # Generate LaTeX tables from results
├── plot_results.py         # Generate drift visualizations
├── COMMUNITY_FINDINGS.md   # Independent validation results
├── docs/                   # Workshop documentation (labs 0-6)
├── harness/                # Core framework code
│   ├── deterministic_retriever.py
│   ├── task_definitions.py
│   └── cross_provider_validation.py
├── providers/              # LLM provider implementations
│   └── watsonx.py          # IBM watsonx.ai integration
├── scripts/                # Data fetching & utilities
│   └── fetch_sec_texts.py  # SEC EDGAR downloader
├── prompts/                # Versioned prompt templates
├── data/                   # Test datasets & generators
├── examples/               # Sample audit trails
└── requirements.txt        # Python dependencies
```

## Reproducibility & Citations

All experiments use **release v0.1.0 (commit c19dac5)** for reproducibility:

```bash
git clone https://github.com/ibm-client-engineering/output-drift-financial-llms
git checkout v0.1.0
```

If you use this framework in your research, please cite:

```bibtex
@article{khatchadourian2025output,
  title={LLM Output Drift: Financial AI Compliance Framework},
  author={Khatchadourian, Raffi and Franco, Rolando},
  journal={arXiv preprint arXiv:2511.07585},
  year={2025}
}
```

**Paper**: [arXiv:2511.07585](https://arxiv.org/abs/2511.07585) | **DOI**: [10.48550/arXiv.2511.07585](https://doi.org/10.48550/arXiv.2511.07585)

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## Contributors & Acknowledgments

This workshop and framework were developed by Raffi Khatchadourian and Rolando Franco in IBM Financial Services in collaboration with researchers focused on responsible AI deployment in regulated industries.

Special thanks to the open-source community and the contributors who helped build and test this framework.

---

!!! success "Ready to Begin?"
    Start with [Lab 0: Workshop Pre-work](pre-work/README.md) to set up your environment!
