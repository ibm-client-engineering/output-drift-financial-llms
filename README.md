# LLM Output Drift: Financial AI Compliance Framework

[![arXiv](https://img.shields.io/badge/arXiv-2601.15322-b31b1b.svg)](https://arxiv.org/abs/2601.15322)
[![arXiv](https://img.shields.io/badge/arXiv-2511.07585-b31b1b.svg)](https://arxiv.org/abs/2511.07585)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Workshop](https://img.shields.io/badge/Workshop-Live-success.svg)](https://ibm-client-engineering.github.io/output-drift-financial-llms/)

> **Key Finding**: 7-20B models achieve 100% deterministic outputs at T=0.0, while 120B+ models exhibit only 12.5-50% consistency—challenging assumptions about model scale for regulated applications.

This framework enables audit-ready AI deployments through deterministic configuration, cross-provider validation, and regulatory-mapped controls for financial services.

**[Interactive Workshop →](https://ibm-client-engineering.github.io/output-drift-financial-llms/)** | Hands-on labs covering setup, experiments, and analysis.

---

## Publications

| Paper | Venue | Focus | Links |
|-------|-------|-------|-------|
| **Replayable Financial Agents** (2026) | [ICLR 2026 FinAI Workshop](https://sites.google.com/view/iclr2026finai/home) (Accepted) | Agent determinism, faithfulness metrics, stress testing | [arXiv:2601.15322](https://arxiv.org/abs/2601.15322) · [DOI](https://doi.org/10.48550/arXiv.2601.15322) |
| **LLM Output Drift** (2025) | [ACM ICAIF 2025 AI4F Workshop](https://ai4f.org/) | Cross-provider validation, model tier classification | [arXiv:2511.07585](https://arxiv.org/abs/2511.07585) · [DOI](https://doi.org/10.48550/arXiv.2511.07585) |

**Code Organization**:
- **Root** (`harness/`, `providers/`, `run_evaluation.py`): Output Drift evaluation framework
- **`econometrics/`**: Replayable Agents extensions—benchmarks, stress testing, econometric modules

---

## Quick Start

```bash
pip install -r requirements.txt
python data/generate_toy_finance.py

# Install Ollama: https://ollama.com/download
ollama pull qwen2.5:7b-instruct
python run_evaluation.py
```

<details>
<summary><strong>Cloud Provider Setup</strong></summary>

**Anthropic (Claude):**
```bash
export ANTHROPIC_API_KEY="your-api-key"
python run_evaluation.py --providers anthropic --models claude-sonnet-4-20250514 --tasks rag
```

**Google (Gemini):**
```bash
export GEMINI_API_KEY="your-api-key"
python run_evaluation.py --providers gemini --models gemini-2.5-pro --tasks rag,sql
```

**IBM watsonx.ai:**
```bash
export WATSONX_API_KEY="your-api-key"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
export WATSONX_PROJECT_ID="your-project-id"
pip install ibm-watsonx-ai>=1.1.0
python run_evaluation.py --providers watsonx --models ibm/granite-3-8b-instruct
```

</details>

<details>
<summary><strong>Fetch Real SEC Data</strong></summary>

```bash
export SEC_USER_AGENT="YourName YourEmail@company.com"
python scripts/fetch_sec_texts.py
# Creates: data/sec/*.txt (used by RAG task)
```

</details>

---

## Model Tiers

Our experiments across 5,185+ runs (480+ non-agentic + 4,705 agentic) reveal **model size inversely correlates with deterministic behavior**:

| Tier | Models | Consistency @ T=0.0 | Compliance |
|------|--------|---------------------|------------|
| **Tier 1** | 7-20B (Granite-3-8B, Qwen2.5-7B, DeepSeek-R1-8B, GPT-OSS-20B) | **94-100%** | ✅ Audit-Ready |
| **Tier 2** | 8-70B cloud (Llama-3.3-70B, Granite-3-8B-watsonx) | 56-100% | △ Task-Specific |
| **Frontier** | Claude Opus 4.5, Claude Sonnet 4, Gemini 2.0 Flash, Gemini 2.5 Pro | **50-96%** | △ Variable |
| **Tier 3** | 120B (GPT-OSS-120B) | **12.5%** | ❌ Non-Compliant |

**Key insight**: Smaller, well-engineered models outperform larger models for regulated financial applications. Frontier models show a **task-structure effect**: 100% SQL determinism but 50-62% RAG consistency. Decision determinism and task accuracy are *not detectably correlated* (r = -0.11, p = 0.63), meaning both must be measured independently.

---

## Navigation

| I want to... | Go to |
|--------------|-------|
| Run drift evaluation | `python run_evaluation.py` |
| Run agent benchmarks | `python -m econometrics.agentic.run_benchmarks` |
| See experimental results | `econometrics/VALIDATION_RESULTS.md` |
| Understand findings | `econometrics/FINDINGS_EXPLAINED.md` |
| Interactive workshop | [Workshop Labs](https://ibm-client-engineering.github.io/output-drift-financial-llms/) |

---

## Framework Components

<details>
<summary><strong>DeterministicRetriever</strong></summary>

SEC 10-K structure-aware retrieval with multi-key ordering that treats retrieval order as a **compliance requirement**.

```python
from harness.deterministic_retriever import DeterministicRetriever

retriever = DeterministicRetriever(corpus_path="data/sec_filings/", chunk_size=512, overlap=50)
results = retriever.retrieve(query="net credit losses 2024", top_k=5)
```

</details>

<details>
<summary><strong>Cross-Provider Validation</strong></summary>

Validates consistency across local (Ollama) and cloud deployments with finance-calibrated invariants (±5% GAAP materiality threshold).

```python
from harness.cross_provider_validation import CrossProviderValidator

validator = CrossProviderValidator(providers=["ollama", "watsonx"], tolerance_pct=5.0)
results = validator.validate(prompt, task_type="sql")
```

</details>

<details>
<summary><strong>Audit Trail System</strong></summary>

Bi-temporal JSONL logging with regulatory mappings (FSB, CFTC).

```python
{
  "timestamp": "2025-11-01T14:23:45Z",
  "model": "granite-3-8b-instruct",
  "temperature": 0.0,
  "seed": 42,
  "prompt_hash": "a3d8f9...",
  "response_hash": "b2c1e7...",
  "compliance_metrics": {"citation_accuracy": 1.0, "schema_valid": true, "decision_flip": false}
}
```

</details>

---

## Repository Structure

| Directory | Purpose |
|-----------|---------|
| `harness/` | Core evaluation framework (retriever, tasks, validation) |
| `providers/` | LLM providers (watsonx, anthropic, gemini) |
| `econometrics/` | Replayable Agents research (benchmarks, metrics, stress tests) |
| `data/` | Synthetic database generation |
| `scripts/` | SEC data fetching utilities |
| `prompts/` | Versioned prompt templates |

---

## Citation

If you use this framework, please cite:

```bibtex
@inproceedings{khatchadourian2026replayable,
  title={Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents},
  author={Khatchadourian, Raffi},
  booktitle={The 2nd ICLR Workshop on Advances in Financial AI (FinAI)},
  year={2026},
  url={https://arxiv.org/abs/2601.15322}
}

@inproceedings{khatchadourian2025output,
  title={LLM Output Drift: Cross-Provider Validation \& Mitigation for Financial Workflows},
  author={Khatchadourian, Raffi and Franco, Rolando},
  booktitle={ACM International Conference on AI in Finance (ICAIF), AI4F Workshop},
  year={2025},
  url={https://arxiv.org/abs/2511.07585}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

This software may be covered by patent applications filed by IBM Corporation. See [NOTICE](NOTICE) for details.

---

**Questions?** Open an issue or contact: raffi.khatchadourian1@ibm.com · rfranco@us.ibm.com

**Acknowledgments**: IBM watsonx.ai, IBM Research, Ollama, Qwen
