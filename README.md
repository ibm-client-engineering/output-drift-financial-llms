# DFAH-Bench

**Replay stability measurement for tool-using AI agents**

DFAH-Bench is a financial AI research artifact and alpha Python package for
measuring whether repeated agent runs preserve both their decisions and their
observable execution paths.

[![DFAH-Bench paper](https://img.shields.io/badge/arXiv-2607.20491-b31b1b.svg)](https://arxiv.org/abs/2607.20491)
[![PyPI](https://img.shields.io/pypi/v/dfah-bench.svg)](https://pypi.org/project/dfah-bench/)
[![Python](https://img.shields.io/pypi/pyversions/dfah-bench.svg)](https://pypi.org/project/dfah-bench/)
[![DFAH package](https://github.com/ibm-client-engineering/output-drift-financial-llms/actions/workflows/dfah-package.yml/badge.svg)](https://github.com/ibm-client-engineering/output-drift-financial-llms/actions/workflows/dfah-package.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Same decision, different path:** a stable final answer can hide a changing
> tool sequence, argument scope, or result. DFAH measures both sides of that
> gap instead of treating the endpoint as the whole system.

DFAH measures repeatability and fidelity to the recorded execution. It does
not inspect hidden reasoning or establish correctness, safety, deployment
readiness, or regulatory compliance.

## Key finding

A separate argument-aware prospective API diagnostic found a much wider range
of trajectory agreement than decision agreement:

| Decision agreement | Exact tool-name path agreement | Strong trajectory agreement |
| ---: | ---: | ---: |
| **94.2–95.1%** | **66.9–69.4%** | **45.0–51.5%** |

Strong trajectory agreement requires exact ordered tool names, canonical
arguments, and deterministic result identities. The diagnostic retained
570/600 episodes across 190 three-replay groups. One predeclared coverage gate
missed by one group, so these aggregates remain diagnostic rather than primary
inference or a provider ranking. A path difference is evidence for review, not
automatically a task failure.

## Quick start: local, free, and no API key

```bash
python -m pip install dfah-bench

dfah check-agent --agent dfah.demo:toy_agent --episode-timeout-s 5
dfah run \
  --agent dfah.demo:toy_agent \
  --replays 3 \
  --episode-timeout-s 5 \
  --out .dfah/runs/toy-local-01
dfah analyze .dfah/runs/toy-local-01 \
  --report .dfah/runs/toy-local-01/report.html
```

No repository clone or API key is needed. The bundled deterministic smoke test
validates the adapter, replay storage, and artifact path. It is not a model-performance
result.

<details>
<summary><strong>Optional virtual environment</strong></summary>

```bash
python -m venv .venv-dfah
source .venv-dfah/bin/activate
python -m pip install dfah-bench
```

Python 3.10 or newer is required.

</details>

### Bring your own agent

```bash
dfah check-agent --agent package.module:agent
```

Use this command first when integrating an existing tool-using agent. Continue
with the [package guide](README_DFAH.md) and the
[bring-your-own-agent walkthrough](docs/dfah/bring-your-own-agent.md).

## Choose a path

| Goal | Start here |
| --- | --- |
| Try DFAH locally | [Run the bundled demo](#quick-start-local-free-and-no-api-key) |
| Test an existing agent | `dfah check-agent --agent package.module:agent` |
| Explore the published results | [Live results explorer](https://ibm-client-engineering.github.io/output-drift-financial-llms/explorer/) |
| Reproduce DFAH-Bench v2 | `make reproduce-paper` and [the reproducibility guide](REPRODUCIBILITY.md) |
| Use the interactive workshop | [Workshop labs](https://ibm-client-engineering.github.io/output-drift-financial-llms/) |

## What DFAH measures

| Measure | Meaning |
| --- | --- |
| Decision Agreement Rate (DAR) | Modal final-decision share within an eligible replay group |
| Tool Agreement Rate (TARseq) | Modal exact ordered tool-name path share on the same denominator |
| Strong trajectory identity | Exact ordered names plus canonical argument and deterministic result hashes, when captured |
| Replay eligibility | Fail-closed qualification of required channels, replay count, and configuration comparability |

Missing or malformed required channels make a replay group ineligible. They are
never converted into agreement, an empty path, or zero divergence. An observed
empty tool path remains valid data. Observable faithfulness means fidelity to
the execution record; it is not a claim about hidden reasoning, accuracy,
materiality, safety, or compliance.

The package also provides versioned suites, resumable append-only episode
storage, review-load reporting, OpenTelemetry GenAI spans, and a pytest plugin.

## Current DFAH-Bench results

The corrected retrospective analysis covers **4,157 episodes from
configurations with observed tool use across 719 comparable replay groups**,
eight retained configurations, and two synthetic financial tasks: compliance
triage and financial DataOps. It is not a model leaderboard.

Among 627 groups with unanimous decisions, 122 (19.5%) changed tool sequence
and 47 (7.5%) changed the set of tools called. Stable decisions can therefore
conceal different observable execution paths.

<details>
<summary><strong>Full model-level results and evidence limitations</strong></summary>

| Coverage | Configuration | Groups | Episodes | Replays | DAR | TARseq | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Complete | Qwen 3.5 | 100 | 800 | 8 | 100.0% | 100.0% | 0.0 pp |
| Complete | Gemma 4 | 100 | 800 | 8 | 99.9% | 99.8% | 0.1 pp |
| Complete | Qwen 2.5 7B | 100 | 800 | 8 | 99.6% | 99.6% | 0.0 pp |
| Complete | GPT-OSS 20B | 100 | 800 | 8 | 98.0% | 97.9% | 0.1 pp |
| Complete | Gemini 2.0 Flash | 100 | 300 | 3 | 93.7% | 88.7% | 5.0 pp |
| Complete, provider default | Claude Sonnet 4 | 100 | 300 | 3 | 94.3% | 73.3% | 21.0 pp |
| Contiguous prefix | Gemini 2.5 Pro | 44 | 132 | 3 | 88.6% | 75.0% | 13.6 pp |
| Contiguous prefix, provider default | Claude Opus 4 | 75 | 225 | 3 | 89.0% | 70.3% | 18.7 pp |

DAR is Decision Agreement Rate; TARseq is exact ordered tool-name agreement.
Rows are task-weighted. The two contiguous-prefix rows have incomplete case
coverage and should not be compared with complete rows as a leaderboard. The
Claude rows use provider-default sampling and are not directly controlled
model comparisons.

The phrase “configurations with observed tool use” is intentional. Twenty-five
retained episodes in nine groups have observed empty tool sequences, while
their configurations used tools elsewhere.

The inconsistent portfolio fixture and its dependent aggregates are excluded.
Historical evidence-contact analysis is also excluded because that channel was
missing not at random. Historical argument and result channels were not
captured and cannot be reconstructed.

The prospective API diagnostic scheduled 600 episodes and retained 570
eligible episodes across 190 groups. Decision agreement was 94.2–95.1%,
ordered name-path agreement was 66.9–69.4%, strong trajectory agreement was
45.0–51.5%, and the separately projected result-only agreement was
54.3–56.9%. One Sonnet/DataOps stratum retained 44 groups against a
predeclared minimum of 45, so the global publication gate did not pass.

Raw prospective provider captures remain approval-gated. The public
reproduction verifies safety-projected aggregate hashes, schemas,
denominators, gates, and published values; it does not recreate provider
calls.

A bounded local systems check retained 792/800 episodes across 99 groups and
repeated its fixed four-tool path exactly. That validates the synthetic harness
and capture path in that setting, not general model determinism or financial
accuracy.

</details>

## Publications

These are three related but distinct studies, not versions of one experiment.

| Paper | Venue | Focus | Links |
| --- | --- | --- | --- |
| **DFAH-Bench: Benchmarking Observable Agent Instability in Financial Decision-Making** (2026) | arXiv | Observable decision and trajectory instability | [arXiv:2607.20491](https://arxiv.org/abs/2607.20491) · [DOI](https://doi.org/10.48550/arXiv.2607.20491) |
| Replayable Financial Agents (2026) | [ICLR 2026 FinAI Workshop](https://sites.google.com/view/iclr2026finai/home) | Agent determinism, faithfulness metrics, and stress testing | [arXiv:2601.15322](https://arxiv.org/abs/2601.15322) · [DOI](https://doi.org/10.48550/arXiv.2601.15322) |
| LLM Output Drift (2025) | [AI4F Workshop 2025](https://ai4f.org/) | Cross-provider validation and mitigation | [arXiv:2511.07585](https://arxiv.org/abs/2511.07585) · [DOI](https://doi.org/10.48550/arXiv.2511.07585) |

## Reproduce the research

The frozen paper artifact and the prospective package are separate surfaces:

- `bench/`, the checked-in replay corpus, and `results/v2/` reproduce the
  DFAH-Bench paper.
- `src/dfah/` is the installable package for new integrations and replay
  captures.

```bash
python -m pip install -r requirements.txt
make reproduce-paper
make verify-v2-manifest
```

The corrected v2 target is offline and reproduces 4,157 episodes and 719
case-level rows from the sanitized public fixture. The archived lineage remains
available separately:

```bash
make reproduce-paper-v1
```

The archived v1 analysis contains 8,127 episodes across 1,338 groups and is
retained for lineage; it is not the corrected-v2 default.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environments, denominator
lineage, eligibility semantics, and limitations. Corrected machine outputs are
under [`results/v2/`](results/v2/).

<details>
<summary><strong>Prior studies and legacy research workflows</strong></summary>

### Earlier Output-Drift Study

An earlier set of experiments grouped tested configurations by their observed
output consistency. These are study-bounded results, not compliance ratings or
a general model-size law:

| Study grouping | Tested configurations | Observed consistency at T=0.0 |
| --- | --- | ---: |
| 7–20B local | Granite-3-8B, Qwen2.5-7B, DeepSeek-R1-8B, GPT-OSS-20B | 94–100% |
| 8–70B cloud | Llama-3.3-70B, Granite-3-8B on watsonx.ai | 56–100% |
| API frontier | Claude Opus 4, Claude Sonnet 4, Gemini 2.0 Flash, Gemini 2.5 Pro | 50–96% |
| 120B local | GPT-OSS-120B | 12.5% |

Repeatability varied by model, task, provider path, and harness. Decision
agreement and task accuracy were not detectably correlated in that study
(r = -0.11, p = 0.63), so they should be measured separately.

### Original output-drift workflow

```bash
python -m pip install -r requirements.txt
python data/generate_toy_finance.py
ollama pull qwen2.5:7b-instruct
python run_evaluation.py
```

Install Ollama from [ollama.com/download](https://ollama.com/download) before
running the local model command.

The stable root commands remain available for the published workshop:

```bash
python run_evaluation.py
python run_dfah_demo.py
python plot_results.py
python make_tables.py
python econometrics/benchmarks/run_all.py
```

Maintained implementations of the four root workshop launchers live under
`scripts/workshop/`. See the [historical harness guide](DFAH.md),
[community findings](COMMUNITY_FINDINGS.md), [agent benchmark guide](econometrics/benchmarks/README.md),
and [econometrics guide](econometrics/README.md).

### Optional provider setup

**Anthropic**

```bash
export ANTHROPIC_API_KEY="your-api-key"
python run_evaluation.py --providers anthropic --models claude-sonnet-4-20250514 --tasks rag
```

**Google Gemini**

```bash
export GEMINI_API_KEY="your-api-key"
python run_evaluation.py --providers gemini --models gemini-2.5-pro --tasks rag,sql
```

**IBM watsonx.ai**

```bash
export WATSONX_API_KEY="your-api-key"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
export WATSONX_PROJECT_ID="your-project-id"
python -m pip install "ibm-watsonx-ai>=1.1.0"
python run_evaluation.py --providers watsonx --models ibm/granite-3-8b-instruct
```

These commands contact external services and may incur cost.

### Optional SEC corpus

```bash
export SEC_USER_AGENT="YourName YourEmail@company.com"
python scripts/fetch_sec_texts.py
```

This networked command populates `data/sec/` for the historical RAG task.

### Historical research API example

```python
from econometrics.agentic.metrics.trajectory_determinism import (
    AgentTrajectory,
    ToolCall,
    analyze_trajectory_determinism,
)

trajectories = [
    AgentTrajectory(
        run_id=f"run_{i}",
        input_context={"alert_id": "TXN-001", "amount": 50000},
        tool_calls=[
            ToolCall(
                tool_name="check_sanctions",
                arguments={"entity": "Acme Corp"},
            )
        ],
        final_decision="escalate",
    )
    for i in range(8)
]
metrics = analyze_trajectory_determinism(trajectories)
print(f"Decision determinism: {metrics.decision_determinism:.1%}")
```

### Historical framework components

The deterministic retriever makes source ordering explicit:

```python
from harness.deterministic_retriever import create_retriever_from_files

retriever = create_retriever_from_files(
    corpus_path="data/sec/",
    chunk_size=200,
    overlap=50,
)
results = retriever.retrieve(query="net credit losses 2024", k=5)
```

Cross-provider validation compares configured deployments. The ±5% tolerance
below is illustrative and is not a universal accounting or compliance
threshold:

```python
from harness.cross_provider_validation import CrossProviderValidator

validator = CrossProviderValidator(
    providers=["ollama", "watsonx"],
    tolerance_pct=5.0,
)
outputs = {"ollama": ollama_result, "watsonx": watsonx_result}
results = validator.validate(outputs, task_type="sql")
```

Historical JSONL records capture selected replay metadata. This example uses a
single event timestamp; it is not a bi-temporal record or regulatory
compliance attestation:

```json
{
  "timestamp": "2025-11-01T14:23:45Z",
  "model": "granite-3-8b-instruct",
  "temperature": 0.0,
  "seed": 42,
  "prompt_hash": "a3d8f9...",
  "response_hash": "b2c1e7...",
  "compliance_metrics": {
    "citation_accuracy": 1.0,
    "schema_valid": true,
    "decision_flip": false
  }
}
```

</details>

<details>
<summary><strong>Repository layout and compatibility paths</strong></summary>

The root retains standard project metadata and documented command launchers.
Removing those launchers would break published labs and external command paths.

| Path | Purpose |
| --- | --- |
| `src/dfah/` | Installable prospective replay package |
| `tests/dfah/` | Package conformance, recovery, privacy, and metric tests |
| `docs/dfah/` | Package quickstart, integration, production, and design guides |
| `bench/` | Frozen metrics and reproduction code for arXiv:2607.20491 |
| `results/v2/` | Corrected v2 paper outputs and release manifest |
| `run_evaluation.py`, `run_dfah_demo.py`, `plot_results.py`, `make_tables.py` | Stable compatibility launchers for published commands |
| `scripts/workshop/` | Maintained implementations behind the root workshop launchers |
| `harness/`, `providers/`, `prompts/`, `data/` | Historical output-drift evaluation components |
| `econometrics/` | Replayable Agents benchmarks, metrics, and stress tests |
| `scripts/` | Reproduction, workshop, and data-fetching utilities |
| `docs/` | Workshop labs, package guides, and results explorer |
| `DFAH.md`, `COMMUNITY_FINDINGS.md` | Historical public documentation paths |

</details>

## Citation

If you use DFAH-Bench, please cite:

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
```

<details>
<summary><strong>Earlier paper citations</strong></summary>

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

</details>

## License

MIT License - See [LICENSE](LICENSE) for details.

This software may be covered by patent applications filed by IBM Corporation.
See [NOTICE](NOTICE) for details.

---

**Questions?** Open an issue or contact: raffi.khatchadourian1@ibm.com ·
rfranco@us.ibm.com

**Acknowledgments**: IBM watsonx.ai, IBM Research, Ollama, Qwen, OpenAI gpt-oss
