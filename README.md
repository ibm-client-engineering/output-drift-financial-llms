# LLM Output Drift: Financial AI Compliance Framework

[![arXiv](https://img.shields.io/badge/arXiv-2601.15322-b31b1b.svg)](https://arxiv.org/abs/2601.15322)
[![arXiv](https://img.shields.io/badge/arXiv-2511.07585-b31b1b.svg)](https://arxiv.org/abs/2511.07585)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Workshop](https://img.shields.io/badge/Workshop-Live-success.svg)](https://ibm-client-engineering.github.io/output-drift-financial-llms/)

> **Key Finding**: 7-20B models achieve 100% deterministic outputs at T=0.0, while 120B+ models exhibit only 12.5-50% consistency—challenging assumptions about model scale for regulated applications.

This framework enables audit-ready AI deployments through deterministic configuration, cross-provider validation, and regulatory-mapped controls for financial services. It includes **[DFAH](DFAH.md)** (Determinism-Faithfulness Assurance Harness), the public harness behind *Replayable Financial Agents* (ICLR 2026).

**[Interactive Workshop →](https://ibm-client-engineering.github.io/output-drift-financial-llms/)** | Hands-on labs covering setup, experiments, and analysis.

---

## Publications

| Paper | Venue | Focus | Links |
|-------|-------|-------|-------|
| **DFAH-Bench: Benchmarking Observable Agent Instability in Financial Decision-Making** (2026) | arXiv preprint (announcement pending) | Replay benchmark: 8,127 episodes, 10 models — outcome-only evaluation misses trajectory/evidence instability | Code: this repo (`bench/`) · `make reproduce-paper` |
| **Replayable Financial Agents** (2026) | [ICLR 2026 FinAI Workshop](https://sites.google.com/view/iclr2026finai/home) | Agent determinism, faithfulness metrics, stress testing | [arXiv:2601.15322](https://arxiv.org/abs/2601.15322) · [DOI](https://doi.org/10.48550/arXiv.2601.15322) |
| **LLM Output Drift** (2025) | [ACM ICAIF 2025 AI4F Workshop](https://ai4f.org/) | Cross-provider validation, model tier classification | [arXiv:2511.07585](https://arxiv.org/abs/2511.07585) · [DOI](https://doi.org/10.48550/arXiv.2511.07585) |

**Code Organization**:
- **Root** (`harness/`, `providers/`, `run_evaluation.py`): Output Drift evaluation framework
- **`econometrics/`**: Replayable Agents extensions—benchmarks, stress testing, econometric modules

---

## Quick Start

```bash
pip install -r requirements.txt

# Try the DFAH demo (no LLM needed, runs in seconds)
python run_dfah_demo.py

# Or run the full output drift evaluation (requires Ollama)
python data/generate_toy_finance.py
ollama pull qwen2.5:7b-instruct   # https://ollama.com/download
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

## DFAH-Bench Results (new paper)

From **DFAH-Bench: Benchmarking Observable Agent Instability in Financial Decision-Making** (arXiv preprint, announcement pending) — 8,127 replay episodes, 10 models, 3 financial tasks. The headline: **outcome-only evaluation reports stable agents whose trajectories and evidence usage are not stable.** Among 912 case groups where decisions agree (DAR ≥ 0.9), 21.8% show trajectory divergence (TAR < 0.9).

| Profile | Model | DAR | TAR | Gap | DCB | ECD | Acc |
|---------|-------|-----|-----|-----|-----|-----|-----|
| Pattern matcher | Qwen 2.5 7B | 0.998 | 0.998 | 0.000 | 0.352 | 0.000 | 33.3% |
| Pattern matcher | Gemma 4 | 0.999 | 0.995 | 0.004 | 0.111 | 0.005 | 56.0% |
| Stable executor | GPT-OSS 20B | 0.963 | 0.956 | 0.007 | 0.132 | 0.017 | 37.3% |
| Stable executor | Gemini 2.0 Flash | 0.953 | 0.891 | 0.062 | 0.143 | 0.081 | 50.7% |
| Trajectory diverger | Gemini 2.5 Pro | 0.860 | 0.747 | 0.113 | 0.099 | 0.186 | 50.2% |
| Trajectory diverger | Claude Opus 4.5 | 0.902 | 0.742 | 0.160 | 0.354 | 0.195 | 44.0% |
| Trajectory diverger | Claude Sonnet 4 | 0.947 | 0.767 | 0.180 | 0.408 | 0.250 | 36.7% |

*DAR = Decision Agreement Rate, TAR = Trajectory Agreement Rate (exact tool-sequence match), Gap = DAR − TAR (the central diagnostic), DCB = Decision Concentration Bias (cross-case, entropy-normalized), ECD = Evidence-Contact Divergence (pairwise Jaccard), Acc = task-weighted accuracy. Full table incl. κ and CIs in the paper.*

Every number regenerates from the raw replay logs in this repo: `make reproduce-paper` (fails loudly on any mismatch; B=10,000 bootstrap, seed=42). Library: `bench/` (metrics, schema, provenance, stats) · Extend to your domain: `examples/domain_extension_medical.py` · Reproducibility details: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## Navigation

| I want to... | Go to |
|--------------|-------|
| Try DFAH (no LLM needed) | `python run_dfah_demo.py` |
| Run drift evaluation (v1) | `python run_evaluation.py` |
| Run agent benchmarks (v2) | `python econometrics/benchmarks/run_all.py` |
| Learn about agent benchmarks | [`econometrics/benchmarks/README.md`](econometrics/benchmarks/README.md) |
| Learn about econometric modules | [`econometrics/README.md`](econometrics/README.md) |
| Interactive workshop | [Workshop Labs](https://ibm-client-engineering.github.io/output-drift-financial-llms/) |

---

## DFAH: Determinism-Faithfulness Assurance Harness

DFAH measures whether your LLM agent produces consistent behavior across repeated runs. It reports **action determinism** (same tools called?), **signature determinism** (same arguments?), **decision determinism** (same final output?), and **accuracy** (correct vs. ground truth).

```bash
python run_dfah_demo.py            # No LLM needed, runs in seconds
# Results saved to dfah_results/dfah_results.json
```

**Use it with your own agent** — see [`examples/dfah_custom_task.py`](examples/dfah_custom_task.py) for a bring-your-own-cases template using the core API:

```python
from econometrics.agentic.metrics.trajectory_determinism import (
    ToolCall, AgentTrajectory, analyze_trajectory_determinism
)

trajectories = [
    AgentTrajectory(
        run_id=f"run_{i}",
        input_context={"alert_id": "TXN-001", "amount": 50000},
        tool_calls=[ToolCall(tool_name="check_sanctions", arguments={"entity": "Acme Corp"})],
        final_decision="escalate",
    )
    for i in range(8)
]
metrics = analyze_trajectory_determinism(trajectories)
print(f"Decision determinism: {metrics.decision_determinism:.1%}")
```

**Full documentation**: [`DFAH.md`](DFAH.md) — output schema, customization guide, benchmark tasks, behavioral profiles.

---

## Framework Components

<details>
<summary><strong>DeterministicRetriever</strong></summary>

SEC 10-K structure-aware retrieval with multi-key ordering that treats retrieval order as a **compliance requirement**.

```python
from harness.deterministic_retriever import create_retriever_from_files

retriever = create_retriever_from_files(corpus_path="data/sec/", chunk_size=200, overlap=50)
results = retriever.retrieve(query="net credit losses 2024", k=5)
```

</details>

<details>
<summary><strong>Cross-Provider Validation</strong></summary>

Validates consistency across local (Ollama) and cloud deployments with finance-calibrated invariants (±5% GAAP materiality threshold).

```python
from harness.cross_provider_validation import CrossProviderValidator

validator = CrossProviderValidator(providers=["ollama", "watsonx"], tolerance_pct=5.0)
outputs = {"ollama": ollama_result, "watsonx": watsonx_result}
results = validator.validate(outputs, task_type="sql")
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

| Path | Purpose |
|------|---------|
| `run_dfah_demo.py` | DFAH entry point (no LLM needed) |
| `DFAH.md` | DFAH documentation, output schema, customization |
| `examples/dfah_custom_task.py` | Bring-your-own-cases template |
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
@article{khatchadourian2026replayable,
  title={Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents},
  author={Khatchadourian, Raffi},
  journal={arXiv preprint arXiv:2601.15322},
  year={2026},
  eprint={2601.15322},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  doi={10.48550/arXiv.2601.15322}
}

@article{khatchadourian2025output,
  title={LLM Output Drift: Cross-Provider Validation \& Mitigation for Financial Workflows},
  author={Khatchadourian, Raffi and Franco, Rolando},
  journal={arXiv preprint arXiv:2511.07585},
  year={2025},
  eprint={2511.07585},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2511.07585}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

This software may be covered by patent applications filed by IBM Corporation. See [NOTICE](NOTICE) for details.

---

**Questions?** Open an issue or contact: raffi.khatchadourian1@ibm.com · rfranco@us.ibm.com

**Acknowledgments**: IBM watsonx.ai, IBM Research, Ollama, Qwen
