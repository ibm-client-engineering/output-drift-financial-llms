# Output Drift and Replay Measurement for Financial AI

[![arXiv](https://img.shields.io/badge/arXiv-2607.20491-b31b1b.svg)](https://arxiv.org/abs/2607.20491)
[![arXiv](https://img.shields.io/badge/arXiv-2601.15322-b31b1b.svg)](https://arxiv.org/abs/2601.15322)
[![arXiv](https://img.shields.io/badge/arXiv-2511.07585-b31b1b.svg)](https://arxiv.org/abs/2511.07585)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Workshop](https://img.shields.io/badge/Workshop-Live-success.svg)](https://ibm-client-engineering.github.io/output-drift-financial-llms/)

> **Same decision, different path:** a stable final answer can hide a changing
> tool sequence, argument scope, or result. DFAH measures both sides of that
> gap instead of treating the endpoint as the whole system.

This repository contains the public replay corpus and analysis code behind the
DFAH-Bench research line, the original output-drift workshop, and an alpha,
pip-installable DFAH package for prospective agent integrations. DFAH measures
repeatability and observable execution fidelity; it does not by itself prove
correctness, safety, or regulatory compliance.

**[Interactive Workshop →](https://ibm-client-engineering.github.io/output-drift-financial-llms/)** ·
**[Live Explorer →](https://ibm-client-engineering.github.io/output-drift-financial-llms/explorer/)** ·
**[DFAH package guide →](README_DFAH.md)**

---

## Publications

| Paper | Venue | Focus | Links |
|-------|-------|-------|-------|
| **DFAH-Bench: Benchmarking Observable Agent Instability in Financial Decision-Making** (2026) | [arXiv:2607.20491](https://arxiv.org/abs/2607.20491) | Repeated decisions can conceal different observable execution paths | [DOI](https://doi.org/10.48550/arXiv.2607.20491) · research API: `bench/` · package: `src/dfah/` |
| **Replayable Financial Agents** (2026) | [ICLR 2026 FinAI Workshop](https://sites.google.com/view/iclr2026finai/home) | Agent determinism, faithfulness metrics, stress testing | [arXiv:2601.15322](https://arxiv.org/abs/2601.15322) · [DOI](https://doi.org/10.48550/arXiv.2601.15322) |
| **LLM Output Drift** (2025) | [AI4F Workshop 2025](https://ai4f.org/) | Cross-provider validation, model tier classification | [arXiv:2511.07585](https://arxiv.org/abs/2511.07585) · [DOI](https://doi.org/10.48550/arXiv.2511.07585) |

**Code Organization**:
- **Root** (`harness/`, `providers/`, `run_evaluation.py`): Output Drift evaluation framework
- **`econometrics/`**: Replayable Agents extensions—benchmarks, stress testing, econometric modules
- **`bench/`**: frozen DFAH-Bench metrics and paper-reproduction pipeline
- **`src/dfah/`**: prospective, pip-installable replay package

---

## Quick Start

### Try the installable DFAH package

```bash
python -m venv .venv-dfah
source .venv-dfah/bin/activate
python -m pip install -e ".[otel]"

dfah check-agent --agent dfah.demo:toy_agent --episode-timeout-s 5
dfah run \
  --agent dfah.demo:toy_agent \
  --replays 3 \
  --episode-timeout-s 5 \
  --out .dfah/runs/toy-local-01
dfah analyze .dfah/runs/toy-local-01 \
  --report .dfah/runs/toy-local-01/report.html
```

This smoke test is deterministic, local, and free. It verifies the adapter and
artifact path before a paid replay; it is not a model-performance result.

### Run the original output-drift workflow

```bash
pip install -r requirements.txt

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

## Earlier Output-Drift Study

An earlier set of experiments grouped the tested configurations by their
observed output consistency. These are study-bounded results, not compliance
ratings or a general model-size law:

| Study grouping | Tested configurations | Observed consistency at T=0.0 |
|------|--------|---------------------|
| 7–20B local | Granite-3-8B, Qwen2.5-7B, DeepSeek-R1-8B, GPT-OSS-20B | 94–100% |
| 8–70B cloud | Llama-3.3-70B, Granite-3-8B on watsonx.ai | 56–100% |
| API frontier | Claude Opus 4, Claude Sonnet 4, Gemini 2.0 Flash, Gemini 2.5 Pro | 50–96% |
| 120B local | GPT-OSS-120B | 12.5% |

The useful operational lesson is narrower: repeatability varied by model,
task, provider path, and harness. Decision agreement and task accuracy were
not detectably correlated in that study (r = -0.11, p = 0.63), so they should
be measured separately.

---

## DFAH-Bench Results

The corrected primary analysis uses **4,157 episodes from configurations with
observed tool use across 719 comparable replay groups**, eight retained
configurations, and two synthetic financial tasks. It is not a model ranking. The point is that
outcome-only evaluation can report a stable agent while its recorded tool path
changes. Among 627 groups with unanimous decisions, 122 (19.5%) changed tool
sequence and 47 (7.5%) changed the set of tools called.

| Coverage | Configuration | Groups | Episodes | Replays | DAR | TARseq | Gap |
|----------|---------------|-------:|---------:|--------:|----:|-------:|----:|
| Complete | Qwen 3.5 | 100 | 800 | 8 | 100.0% | 100.0% | 0.0 pp |
| Complete | Gemma 4 | 100 | 800 | 8 | 99.9% | 99.8% | 0.1 pp |
| Complete | Qwen 2.5 7B | 100 | 800 | 8 | 99.6% | 99.6% | 0.0 pp |
| Complete | GPT-OSS 20B | 100 | 800 | 8 | 98.0% | 97.9% | 0.1 pp |
| Complete | Gemini 2.0 Flash | 100 | 300 | 3 | 93.7% | 88.7% | 5.0 pp |
| Complete, provider default | Claude Sonnet 4 | 100 | 300 | 3 | 94.3% | 73.3% | 21.0 pp |
| Contiguous prefix | Gemini 2.5 Pro | 44 | 132 | 3 | 88.6% | 75.0% | 13.6 pp |
| Contiguous prefix, provider default | Claude Opus 4 | 75 | 225 | 3 | 89.0% | 70.3% | 18.7 pp |

DAR is Decision Agreement Rate; TARseq is exact ordered tool-name agreement.
Rows are task-weighted. The two prefix rows have incomplete case coverage and
should not be compared with complete rows as a leaderboard. Missing or
malformed required channels make a replay group ineligible; they are never
scored as agreement or zero divergence.

A separate argument-aware API diagnostic retained 570/600 episodes across 190
groups. DAR was 94.2–95.1%; ordered name-path agreement was 66.9–69.4%;
name-plus-canonical-argument agreement was 45.0–51.5%; and result-only
agreement was 54.3–56.9%. One predeclared coverage gate missed by one group,
so those aggregates remain diagnostic. A bounded local
systems check retained 792/800 episodes across 99 groups and repeated its
fixed four-tool path exactly; that validates the harness in that setting, not
general model determinism or accuracy.

The raw ledger remains available for lineage. `make reproduce-paper`
regenerates the corrected v2 retrospective artifacts and validates the
aggregate-only extensions and release manifest. The explicit
`make reproduce-paper-v1` target reconstructs the archived public v1 analysis.
Package: `src/dfah/` · Historical research API: `bench/` · Reproducibility:
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) · Corrected machine outputs:
[`results/v2/`](results/v2/).

---

## Navigation

| I want to... | Go to |
|--------------|-------|
| Qualify an agent adapter | `dfah check-agent --agent package.module:agent` |
| Run the installed no-network demo | `dfah run --agent dfah.demo:toy_agent --replays 3` |
| Reproduce the corrected v2 analysis | `make reproduce-paper` |
| Reproduce the archived v1 replay analysis | `make reproduce-paper-v1` |
| Run drift evaluation (v1) | `python run_evaluation.py` |
| Run agent benchmarks (v2) | `python econometrics/benchmarks/run_all.py` |
| Learn about agent benchmarks | [`econometrics/benchmarks/README.md`](econometrics/benchmarks/README.md) |
| Learn about econometric modules | [`econometrics/README.md`](econometrics/README.md) |
| Interactive workshop | [Workshop Labs](https://ibm-client-engineering.github.io/output-drift-financial-llms/) |

---

## DFAH: Determinism-Faithfulness Assurance Harness

DFAH uses “faithfulness” in a deliberately observable sense: replay fidelity
to the execution record. It asks whether comparable runs preserve the
decision, tool sequence, and—when captured—canonical argument and result
identities. It does not inspect hidden reasoning or substitute for accuracy
evaluation.

```bash
dfah check-agent --agent dfah.demo:toy_agent
dfah run --agent dfah.demo:toy_agent --replays 3
```

The current package adds fail-closed replay eligibility, versioned suites,
resumable episode storage, review-load reporting, OpenTelemetry GenAI spans,
and a pytest plugin. Start with the [package guide](README_DFAH.md) or the
[bring-your-own-agent walkthrough](docs/dfah/bring-your-own-agent.md).

The older research API remains available for exact reproduction. See
[`examples/dfah_custom_task.py`](examples/dfah_custom_task.py):

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

**Historical harness documentation**: [`DFAH.md`](DFAH.md) ·
**Prospective package documentation**: [`README_DFAH.md`](README_DFAH.md)

---

## Framework Components

<details>
<summary><strong>DeterministicRetriever</strong></summary>

SEC 10-K structure-aware retrieval with multi-key ordering that makes retrieval
order explicit and reproducible for downstream review.

```python
from harness.deterministic_retriever import create_retriever_from_files

retriever = create_retriever_from_files(corpus_path="data/sec/", chunk_size=200, overlap=50)
results = retriever.retrieve(query="net credit losses 2024", k=5)
```

</details>

<details>
<summary><strong>Cross-Provider Validation</strong></summary>

Compares consistency across local (Ollama) and cloud deployments. The example
uses a configurable ±5% numeric tolerance for demonstration; it is not a
universal accounting or compliance threshold.

```python
from harness.cross_provider_validation import CrossProviderValidator

validator = CrossProviderValidator(providers=["ollama", "watsonx"], tolerance_pct=5.0)
outputs = {"ollama": ollama_result, "watsonx": watsonx_result}
results = validator.validate(outputs, task_type="sql")
```

</details>

<details>
<summary><strong>Replay Record System</strong></summary>

JSONL records with timestamps, model settings, input/output hashes, and
descriptive validation fields. This sample uses one event timestamp; it is not
a bi-temporal record or a regulatory compliance attestation.

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
| `src/dfah/` | Installable prospective replay package |
| `tests/dfah/` | Package conformance, recovery, privacy, and metric tests |
| `docs/dfah/` | Package quickstart, integration, production, and design guides |
| `bench/` | Frozen metrics and reproduction code for arXiv:2607.20491 |
| `run_dfah_demo.py` | Historical DFAH demo |
| `DFAH.md` | Historical harness documentation |
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
@article{khatchadourian2026dfahbench,
  title={DFAH-Bench: Benchmarking Observable Agent Instability in Financial Decision-Making},
  author={Khatchadourian, Raffi},
  journal={arXiv preprint arXiv:2607.20491},
  year={2026},
  eprint={2607.20491},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  doi={10.48550/arXiv.2607.20491},
  url={https://arxiv.org/abs/2607.20491}
}

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

@inproceedings{khatchadourian2025output,
  title={LLM Output Drift: Cross-Provider Validation \& Mitigation for Financial Workflows},
  author={Khatchadourian, Raffi and Franco, Rolando},
  booktitle={AI4F Workshop},
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

**Acknowledgments**: IBM watsonx.ai, IBM Research, Ollama, Qwen, OpenAI gpt-oss
