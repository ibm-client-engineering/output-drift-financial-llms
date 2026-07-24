# Research Papers

## Publications

| Paper | Venue | Year |
|-------|-------|------|
| [Same Decision, Different Path: DFAH-Bench for AI Agents in Finance](https://arxiv.org/abs/2607.20491) | [arXiv:2607.20491](https://arxiv.org/abs/2607.20491) | 2026 |
| [Replayable Financial Agents](https://arxiv.org/abs/2601.15322) | [ICLR 2026 FinAI Workshop](https://sites.google.com/view/iclr2026finai/home) | 2026 |
| [LLM Output Drift](https://arxiv.org/abs/2511.07585) | [AI4F Workshop 2025](https://ai4f-workshop.github.io/) | 2025 |

---

## Same Decision, Different Path: DFAH-Bench for AI Agents in Finance (2026)

**Paper**: [arXiv:2607.20491](https://arxiv.org/abs/2607.20491) |
**DOI**: [10.48550/arXiv.2607.20491](https://doi.org/10.48550/arXiv.2607.20491)

The newest paper in this research line measures **whether agents that agree on
decisions also agree on the recorded path to those decisions**. The corrected
primary analysis uses 4,157 episodes from configurations with observed tool
use across 719 comparable groups, eight configurations, and two synthetic
financial tasks.

**Key result**: 122 of 627 unanimous-decision groups (19.5%) changed tool
sequence, and 47 (7.5%) changed the tool-name set. A separate prospective API
diagnostic found 94.2–95.1% decision agreement but only 66.9–69.4% exact
name-path agreement. The benchmark measures repeatability and observable
execution fidelity, not correctness.

The default command regenerates and verifies the corrected v2 release:

```bash
make reproduce-paper
make reproduce-paper-v1  # archived lineage only
```

See [Lab 8](../lab-8/README.md) for the guided walkthrough and the
[README results table](https://github.com/ibm-client-engineering/output-drift-financial-llms#dfah-bench-results-new-paper)
for the package and corrected-analysis walkthrough.

---

## Earlier study: LLM Output Drift

This page summarizes the key findings from our research papers on output drift and agent determinism in large language models used for financial applications.

---

## The Core Problem

Large Language Models (LLMs) can exhibit **output drift**: the same prompt can
produce different outputs across repeated runs, even at temperature=0.0. In
regulated workflows, that variation creates additional validation,
documentation, and monitoring work.

**The Question**: Can smaller models be more reliable than larger ones for deterministic, compliance-critical tasks?

---

## The Counterintuitive Finding

### A model-size pattern in the tested conditions

Our research reveals a **counterintuitive result**:

- **7-20B tested configurations**: up to **100% observed output consistency** at temperature=0.0
- **The tested 120B configuration**: **12.5% observed consistency [95% CI: 3.5–36.0%]**

This challenges the conventional wisdom that "bigger is always better" in AI systems.

!!! info "Statistical Notation Used in This Paper"
    Throughout our findings, we report:

    - **95% Confidence Interval (CI)**: The range within which we are 95% confident the true consistency rate lies. For example, "12.5% [3.5–36.0]" means the measured consistency was 12.5%, but the true value likely falls between 3.5% and 36.0%.
    - **𝑝-value**: Measures whether differences between models are statistically significant. Values 𝑝 < 0.05 indicate significance; 𝑝 < 0.0001 indicates highly significant differences unlikely due to chance.

    All Tier 1 vs Tier 3 comparisons showed 𝑝 < 0.0001, indicating the performance differences are highly statistically significant.

### Why This Matters

The result motivates measuring the exact deployed model, task, provider path,
and harness rather than inferring repeatability from parameter count.

---

## Methodology

### Experimental Design

- **Models Tested**:
  - Tier 1 (7-20B): Qwen2.5-7B, IBM Granite-3-8B, GPT-OSS-20B
  - Tier 2 (40-70B): Llama-3.3-70B, Mistral-Medium
  - Tier 3 (120B+): GPT-OSS-120B

- **Total Runs**: 480 experiments (n=16 concurrent runs per condition)
- **Tasks**: SQL generation, RAG Q&A, JSON summarization
- **Providers**: Ollama (local), IBM watsonx.ai (cloud), Anthropic, Google Gemini
- **Key Parameters**: temperature=0.0, seed=42 (deterministic settings)

### Reproducibility

All experiments are reproducible using release v0.1.0:

```bash
git clone https://github.com/ibm-client-engineering/output-drift-financial-llms
cd output-drift-financial-llms
pip install -r requirements.txt
python run_evaluation.py \
  --models qwen2.5:7b-instruct,granite-3-8b,llama-3.3-70b \
  --temperatures 0.0,0.2 \
  --concurrency 1,4,16 \
  --repeats 16
```

---

## Key Findings

### 1. Study groupings

Based on output consistency at temperature=0.0:

| Study label | Tested configurations | Observed consistency | Interpretation |
|------|--------|-------------|-------------------|
| **Group 1** | 7-20B (Qwen2.5-7B, Granite-3-8B, GPT-OSS-20B) | up to **100%** | High repeatability in the tested condition |
| **Group 2** | 40-70B (Llama-3.3-70B, Mistral-Medium) | 56-100% | Task-dependent in this sample |
| **Group 3** | 120B+ (GPT-OSS-120B) | **12.5%** | Low repeatability in this condition |

**Interpretation**:
- These are bounded observations, not deployment or compliance ratings.
- Validate the exact model, provider, task, prompt, and harness configuration.
- Repeatability and correctness need separate measurements.

### 2. Task-Specific Results (Temperature=0.0)

| Task Type | Tier 1 (7-20B) | Tier 2 (40-70B) | Tier 3 (120B) |
|-----------|---------------|-----------------|---------------|
| **SQL Generation** | 100% | 100% | 12.5% |
| **Summarization** | 100% | 87.5% | 12.5% |
| **RAG Q&A** | 93.75% | 75.0% | 12.5% |

**Key Insight**: The tested less-structured RAG task retained more than 90%
consistency for the highest-repeatability study group.

### 3. Temperature Sensitivity

RAG task consistency as temperature increases:

| Temperature | Qwen2.5-7B (Tier 1) | Llama-3.3-70B (Tier 2) | GPT-OSS-120B (Tier 3) |
|-------------|---------------------|------------------------|------------------------|
| **T=0.0** | 93.75% | 75.0% | 12.5% |
| **T=0.2** | 56.25% | 43.75% | 6.25% |
| **T=1.0** | 18.75% | 12.5% | 0% |

**Takeaway**: In this experiment, moving from 0.0 to 0.2 increased drift.
Record the setting and test it, but do not treat temperature 0 as a guarantee.

### 4. Cross-Provider Validation

Testing Tier 1 model consistency across providers:

| Provider Pair | Model | Consistency | Validated |
|---------------|-------|-------------|-----------|
| Ollama ↔ watsonx.ai | Qwen2.5-7B → Granite-3-8B | ≥95% | ✅ |
| Ollama ↔ watsonx.ai | Granite-3-8B → Granite-3-8B | 100% | ✅ |
| Ollama ↔ OpenAI | Qwen2.5-7B → GPT-4 | <50% | ❌ |

**Finding**: Some tested local/cloud pairs had high agreement. A migration
still requires validation of the actual serving stack and workload.

### 5. Regulatory Alignment

The framework can contribute evidence to broader governance processes:

| Regulation | Requirement | Framework Solution |
|------------|-------------|--------------------|
| **SR 11-7** | Model validation & ongoing monitoring | Versioned replay records |
| **ECOA / FCRA** | Decision documentation | Reconstructable run artifacts |
| **GDPR Art. 22** | Review of automated decisions | Decision and tool-path capture |
| **FSB** | Third-party model risk | Configuration-specific comparison |

These mappings are design aids, not a certification of regulatory compliance.

---

## Technical Innovations

### 1. DeterministicRetriever

Ensures reproducible SEC 10-K retrieval with multi-key ordering:

- **Problem**: Standard RAG systems use non-deterministic vector similarity
- **Solution**: Multi-level sorting (score → document_id → chunk_id) ensures identical results
- **Benefit**: Same query always returns same chunks in same order

### 2. CrossProviderValidator

Validates consistency across deployment environments:

- **Problem**: Models behave differently on different infrastructure
- **Workshop implementation**: Automated comparison with a configurable numeric
  tolerance (±5% in the example; not a universal GAAP threshold)
- **Benefit**: Measure behavior before and after a deployment change

### 3. Replay Records

JSONL format capturing:

- Input prompt + response hashes (SHA-256)
- Model parameters (temperature, seed, version)
- Descriptive checks (schema validity, source-reference matching)
- Workshop metadata for later governance review

The example contains a single event timestamp. It does not implement
bi-temporal data modeling or determine compliance with a named rule.

---

## Practical Implications

### For Financial Institutions

1. **Vendor Selection**: Measure the workload, not parameter count alone
2. **Decoding Policy**: Pin and record decoding settings
3. **Model Validation**: Compare the exact provider stacks before a change
4. **Run Records**: Retain inputs, outputs, versions, and observable tool paths

### For Model Developers

1. **Architecture**: Optimize for determinism, not just accuracy
2. **Testing**: Include multi-run consistency metrics in benchmarks
3. **Documentation**: Report consistency scores alongside performance metrics

### For Regulators

1. **Standards**: Define risk-based repeatability and review thresholds
2. **Validation**: Require cross-provider equivalence testing
3. **Monitoring**: Mandate ongoing drift detection in production

---

## Limitations

1. **Model Scope**: Tested 5 models; findings may not generalize to all architectures
2. **Task Coverage**: Focused on SQL, RAG, summarization—other tasks (e.g., generation) may differ
3. **Infrastructure**: Results specific to tested providers (Ollama, watsonx.ai)
4. **Temporal Stability**: Long-term consistency (months/years) not evaluated

---

## Future Work

1. **Expanded Model Coverage**: Test emerging architectures (Gemma-2-9B, Phi-4, etc.)
2. **Additional Tasks**: Credit risk, fraud detection, portfolio optimization
3. **Regulatory Integration**: Pilot with partner banks under SR 11-7 supervision
4. **Drift Mitigation**: Techniques to improve Tier 2/3 consistency

---

## Citation

If you use this framework or findings in your research, please cite:

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
  booktitle={AI4F Workshop},
  year={2025},
  url={https://arxiv.org/abs/2511.07585}
}
```

**DFAH-Bench**: [arXiv:2607.20491](https://arxiv.org/abs/2607.20491) | **DOI**: [10.48550/arXiv.2607.20491](https://doi.org/10.48550/arXiv.2607.20491)
**Replayable Agents**: [arXiv:2601.15322](https://arxiv.org/abs/2601.15322) | **DOI**: [10.48550/arXiv.2601.15322](https://doi.org/10.48550/arXiv.2601.15322)
**Output Drift**: [arXiv:2511.07585](https://arxiv.org/abs/2511.07585) | **DOI**: [10.48550/arXiv.2511.07585](https://doi.org/10.48550/arXiv.2511.07585)

---

## Related Resources

- **Full Paper**: [arXiv:2511.07585](https://arxiv.org/abs/2511.07585)
- **Code Repository**: [GitHub](https://github.com/ibm-client-engineering/output-drift-financial-llms)
- **API Documentation**: [API Reference](api.md)
- **Workshop Labs**: [Lab 0-7](../lab-1/README.md)

---

## Key Takeaways

1. **Size is not a repeatability guarantee**: test the exact configuration
2. **Temperature 0 is not a determinism guarantee**: pin and measure settings
3. **Repeatability is not correctness or compliance**
4. **Provider changes need replay validation**
5. **The framework is open source and extensible**

---

**Questions?** See [Troubleshooting Guide](troubleshooting.md) or open an issue on [GitHub](https://github.com/ibm-client-engineering/output-drift-financial-llms/issues).
