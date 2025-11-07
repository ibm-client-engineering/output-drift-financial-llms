# LLM Output Drift: Financial AI Compliance Framework

[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Workshop](https://img.shields.io/badge/Workshop-Live-success.svg)](https://ibm-client-engineering.github.io/output-drift-financial-llms/)

> **Key Finding**: 7-8B models achieve 100% deterministic outputs at T=0.0, while 120B models exhibit only 12.5% consistency—fundamentally challenging assumptions about model scale for regulated applications.

**📚 [Interactive Workshop →](https://ibm-client-engineering.github.io/output-drift-financial-llms/)** | Complete hands-on labs (0-6) covering setup, experiments, analysis, and framework extension.

This repository contains the evaluation framework from our ACM ICAIF 2025 paper demonstrating how to achieve audit-ready AI deployments through deterministic configuration, cross-provider validation, and regulatory-mapped controls.

## 🎯 Quick Start (5 Minutes)

```bash
# Install dependencies
pip install -r requirements.txt
```

```bash
# Generate synthetic financial database
python data/generate_toy_finance.py
```

```bash
# First, install Ollama: https://ollama.com/download
ollama pull qwen2.5:7b-instruct
```

```bash
# Run deterministic evaluation (requires Ollama)
python run_evaluation.py
```

**Don't have Ollama?** Install from https://ollama.com/download (macOS, Linux, Windows)

## 📊 Model Tiers for Financial Compliance

Our experiments across 480 runs (n=16 per condition) reveal **model size inversely correlates with deterministic behavior**:

| Tier | Models | Consistency @ T=0.0 | Compliance | Recommended Use |
|------|--------|---------------------|------------|-----------------|
| **Tier 1** | 7-8B (Granite-3-8B, Qwen2.5-7B) | **100%** | ✅ Audit-Ready | **All regulated tasks** |
| **Tier 2** | 40-70B (Llama-3.3-70B, Mistral) | 56-100% | △ Task-Specific | SQL/structured only |
| **Tier 3** | 120B (GPT-OSS-120B) | **12.5%** | ❌ Non-Compliant | **Avoid for compliance** |

**Key insight**: Smaller, well-engineered models (7-8B) outperform larger models (120B+) for regulated financial applications. Granite-3-8B and Qwen2.5-7B achieve perfect output consistency required for audit trails, while GPT-OSS-120B's 12.5% consistency makes it unsuitable for credit decisions, regulatory reporting, or any workflow requiring reproducibility.

## 🔧 Framework Components

### 1. DeterministicRetriever
SEC 10-K structure-aware retrieval with multi-key ordering (score↓, section\_priority↑, snippet\_id↑, chunk\_idx↑) that treats retrieval order as a **compliance requirement** rather than a performance optimization.

```python
from harness.deterministic_retriever import DeterministicRetriever

retriever = DeterministicRetriever(
    corpus_path="data/sec_filings/",
    chunk_size=512,
    overlap=50
)
results = retriever.retrieve(query="net credit losses 2023", top_k=5)
# Returns deterministic, ordered chunks with stable IDs
```

### 2. Cross-Provider Validation
Validates consistency across local (Ollama) and cloud (IBM watsonx.ai) deployments with finance-calibrated invariants (±5% materiality threshold from GAAP auditing standards).

```python
from harness.cross_provider_validation import CrossProviderValidator

validator = CrossProviderValidator(
    providers=["ollama", "watsonx"],
    tolerance_pct=5.0  # GAAP materiality threshold
)
results = validator.validate(prompt, task_type="sql")
# Returns: consistency=True if outputs match within 5%
```

### 3. Audit Trail System
Bi-temporal JSONL logging with regulatory mappings (FSB "consistent decisions", CFTC "document AI outcomes").

```python
# Example audit trail entry
{
  "timestamp": "2025-11-01T14:23:45Z",
  "model": "granite-3-8b-instruct",
  "temperature": 0.0,
  "seed": 42,
  "prompt_hash": "a3d8f9...",
  "response_hash": "b2c1e7...",
  "citations": ["citi_2024_10k", "gs_2024_10k"],
  "compliance_metrics": {
    "citation_accuracy": 1.0,
    "schema_valid": true,
    "decision_flip": false
  }
}
```

## 📁 Repository Contents

- **`harness/`**: Core deterministic evaluation framework
  - `deterministic_retriever.py`: SEC-aware retrieval with stable ordering
  - `task_definitions.py`: RAG, SQL, and JSON summarization tasks
  - `cross_provider_validation.py`: Multi-provider consistency gates
- **`data/`**: Synthetic database generation scripts
- **`prompts/`**: Versioned prompt templates from paper (Appendix D)
- **`examples/`**: Sample audit trails demonstrating regulatory traceability

## 🔬 Reproducing Paper Results

Our experiments evaluated 5 models across 480 runs (n=16 per condition):

```bash
# Run full evaluation suite
python run_evaluation.py \
  --models qwen2.5:7b granite-3-8b llama-3.3-70b \
  --temperatures 0.0 0.2 \
  --concurrency 1,4,16 \
  --output traces/
```

Results will be saved as JSONL traces in `traces/*.jsonl` with complete reproducibility manifests.

## 🚀 Deployment Guidance

**For Tier 1 models (Granite-3-8B, Qwen2.5:7B):**
- ✅ All regulated tasks: credit decisions, regulatory reporting, client communications
- ✅ Audit-ready: 100% deterministic at T=0.0
- ✅ Cross-provider validated: local (Ollama) and cloud (IBM watsonx.ai)

**For Tier 2 models (Llama-3.3-70B, Mistral-Medium):**
- △ SQL/structured tasks only (100% consistency)
- ❌ Avoid RAG tasks (56% consistency at T=0.0)

**For Tier 3 models (GPT-OSS-120B):**
- ❌ Not suitable for regulated financial applications
- Consistency: 12.5% across all tasks and temperatures

## 🏢 Data Sources

- **SEC 10-K Filings**: Citigroup, Goldman Sachs, JPMorgan Chase (2024) from [EDGAR](https://www.sec.gov/edgar/search/)
- **Synthetic Database**: Generated via Faker library (see `data/generate_toy_finance.py`)
- **Prompt Templates**: Complete templates in `prompts/templates.json`

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🔒 Notice

This software may be covered by one or more patent applications filed by IBM Corporation. The MIT license does not include patent grants. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for complete details.

For patent licensing inquiries, contact IBM Corporation's Intellectual Property Law Department.

## 🙏 Acknowledgments

IBM watsonx.ai, IBM Research, and the open-source communities behind Ollama and Qwen.

---

**Questions?** Open an issue or contact: raffi.khatchadourian1@ibm.com or rfranco@us.ibm.com

**Paper**: [arXiv:2025.XXXXX](https://arxiv.org/abs/XXXXX) | **Repository**: [http://github.com/ibm-client-engineering/output-drift-financial-llms](http://github.com/ibm-client-engineering/output-drift-financial-llms)
