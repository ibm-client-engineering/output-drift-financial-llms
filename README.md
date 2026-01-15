# LLM Output Drift: Financial AI Compliance Framework

[![arXiv](https://img.shields.io/badge/arXiv-2511.07585-b31b1b.svg)](https://arxiv.org/abs/2511.07585)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Workshop](https://img.shields.io/badge/Workshop-Live-success.svg)](https://ibm-client-engineering.github.io/output-drift-financial-llms/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Framework-blue)](https://huggingface.co/raffi-souren/llm-output-drift-financial-workflows)

> **Key Finding**: 7-20B models achieve 100% deterministic outputs at T=0.0, while 120B models exhibit only 12.5% consistency—fundamentally challenging assumptions about model scale for regulated applications.

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

### Optional: Cloud Provider Setup

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

### Optional: Fetch Real SEC Data

By default, the framework uses synthetic data. To fetch real SEC 10-K filings:

```bash
# Set SEC User-Agent (required by SEC EDGAR)
export SEC_USER_AGENT="YourName YourEmail@company.com"

# Download 2024 10-K filings for Citigroup, Goldman Sachs, JPMorgan
python scripts/fetch_sec_texts.py

# Creates: data/sec/*.txt (used by RAG task)
```

## 📊 Model Tiers for Financial Compliance

Our experiments across 480+ runs (n=16 per condition) reveal **model size inversely correlates with deterministic behavior**:

| Tier | Models | Consistency @ T=0.0 | Compliance | Recommended Use |
|------|--------|---------------------|------------|-----------------|
| **Tier 1** | 7-20B (Granite-3-8B, Qwen2.5-7B, GPT-OSS-20B, Claude Sonnet 4) | **100%** | ✅ Audit-Ready | **All regulated tasks** |
| **Tier 2** | 40-70B (Llama-3.3-70B, Mistral) | 56-100% | △ Task-Specific | SQL/structured only |
| **Frontier** | 120B+ (Claude Opus 4.5, Gemini 2.5 Pro) | **50-100%** | △ Task-Dependent | SQL only (100%), avoid RAG |
| **Tier 3** | 120B (GPT-OSS-120B) | **12.5%** | ❌ Non-Compliant | **Avoid for compliance** |

**Key insight**: Smaller, well-engineered models (7-20B) outperform larger models (120B+) for regulated financial applications. Frontier models (Claude Opus, Gemini) show a **task-structure effect**: 100% SQL determinism but only 50-62% RAG consistency under load. This suggests schema constraints enforce determinism while unstructured generation exposes architectural non-determinism.

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
results = retriever.retrieve(query="net credit losses 2024", top_k=5)
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
- **`providers/`**: LLM provider implementations
  - `watsonx.py`: IBM watsonx.ai cloud provider (requires API credentials)
  - `anthropic.py`: Anthropic Claude API (requires `ANTHROPIC_API_KEY`)
  - `gemini.py`: Google Gemini API (requires `GEMINI_API_KEY`)
- **`scripts/`**: Data acquisition and utilities
  - `fetch_sec_texts.py`: Download SEC 10-K filings from EDGAR
- **`data/`**: Synthetic database generation scripts
  - `generate_toy_finance.py`: Create SQLite database for SQL tasks
- **`prompts/`**: Versioned prompt templates from paper (Appendix D)
- **`examples/`**: Sample audit trails demonstrating regulatory traceability
- **`make_tables.py`**: Generate LaTeX tables from results (for paper reproduction)
- **`plot_results.py`**: Create visualizations from aggregate statistics
- **`run_evaluation.py`**: Main orchestrator supporting Ollama and watsonx providers
- **`econometrics/`**: v2 Replayable Agents experiments
  - `agentic/`: Trajectory determinism & faithfulness metrics
  - `benchmarks/`: 3 financial tasks × 50 test cases each

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

### Analysis and Visualization

After running evaluations, generate visualizations from your results:

```bash
# Create drift analysis plots (requires results/aggregate.csv)
python plot_results.py
# Output: figs/*.png (drift vs concurrency, latency, etc.)
```

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

## 📂 About This Repository

**Community-focused, fully-featured framework** for LLM output drift evaluation in financial applications.

**Key Features**:
- **Ollama-first**: Local evaluation without cloud dependencies
- **Multi-provider**: Optional IBM watsonx.ai and other cloud providers
- **Complete tooling**: Analysis, visualization, and SEC data fetching
- **Workshop materials**: Interactive labs for hands-on learning

**Development**: Maintained by IBM with community contributions. We welcome issues, pull requests, and validation results (see `COMMUNITY_FINDINGS.md`).

## 📚 Paper Lineage

This repository supports two related research papers:

| Paper | Venue | Focus | Code Location |
|-------|-------|-------|---------------|
| **LLM Output Drift** | ACM ICAIF 2025 | Cross-provider determinism validation | Root folder (`harness/`, `providers/`) |
| **Replayable Financial Agents** | *Upcoming* | Agent trajectory determinism & faithfulness | `econometrics/` folder |

**v1 → v2 Evolution**: The original work measured output consistency for single-turn tasks (SQL, RAG, summarization). The follow-up extends to **tool-using agents**, introducing trajectory determinism metrics and stress-test harnesses for production disruptions.

> 📖 Workshop materials in `docs/` are for the v1 paper only.

## 📄 Citation

If you use this framework in your research, please cite our paper:

```bibtex
@article{khatchadourian2025output,
  title={LLM Output Drift: Financial AI Compliance Framework},
  author={Khatchadourian, Raffi and Franco, Rolando},
  journal={arXiv preprint arXiv:2511.07585},
  year={2025}
}
```

**Paper**: [arXiv:2511.07585](https://arxiv.org/abs/2511.07585) | **DOI**: [10.48550/arXiv.2511.07585](https://doi.org/10.48550/arXiv.2511.07585)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🔒 Notice

This software may be covered by one or more patent applications filed by IBM Corporation. The MIT license does not include patent grants. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for complete details.

For patent licensing inquiries, contact IBM Corporation's Intellectual Property Law Department.

## 🙏 Acknowledgments

IBM watsonx.ai, IBM Research, and the open-source communities behind Ollama and Qwen.

---

**Questions?** Open an issue or contact: raffi.khatchadourian1@ibm.com or rfranco@us.ibm.com

**Paper**: [arXiv:2511.07585](https://arxiv.org/abs/2511.07585) | **Repository**: [http://github.com/ibm-client-engineering/output-drift-financial-llms](http://github.com/ibm-client-engineering/output-drift-financial-llms)
