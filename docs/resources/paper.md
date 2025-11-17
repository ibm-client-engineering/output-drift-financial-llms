# Research Paper Summary

## LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows

This page summarizes the key findings from our research paper on output drift in large language models used for financial applications.

---

## The Core Problem

Large Language Models (LLMs) exhibit **output drift**: non-deterministic behavior where the same prompt produces different outputs across multiple runs, even at temperature=0.0. For financial institutions subject to regulations like SR 11-7 (Model Risk Management), ECOA, and GDPR, this creates significant compliance risks.

**The Question**: Can smaller models be more reliable than larger ones for deterministic, compliance-critical tasks?

---

## The Counterintuitive Finding

### Smaller Models Win for Determinism

Our research reveals a **counterintuitive result**:

- **7-8B parameter models**: Achieve **100% output consistency** at temperature=0.0
- **120B parameter models**: Only **12.5% consistency [95% CI: 3.5–36.0%]** under identical conditions

This challenges the conventional wisdom that "bigger is always better" in AI systems.

!!! info "Statistical Notation Used in This Paper"
    Throughout our findings, we report:

    - **95% Confidence Interval (CI)**: The range within which we are 95% confident the true consistency rate lies. For example, "12.5% [3.5–36.0]" means the measured consistency was 12.5%, but the true value likely falls between 3.5% and 36.0%.
    - **𝑝-value**: Measures whether differences between models are statistically significant. Values 𝑝 < 0.05 indicate significance; 𝑝 < 0.0001 indicates highly significant differences unlikely due to chance.

    All Tier 1 vs Tier 3 comparisons showed 𝑝 < 0.0001, indicating the performance differences are highly statistically significant.

### Why This Matters

For regulated financial applications requiring **reproducible audit trails**, smaller models are not just adequate—they are **superior** to larger models when deterministic behavior is required.

---

## Methodology

### Experimental Design

- **Models Tested**:
  - Tier 1 (7-8B): Qwen2.5-7B, IBM Granite-3-8B
  - Tier 2 (40-70B): Llama-3.3-70B, Mistral-Medium
  - Tier 3 (120B+): GPT-OSS-120B

- **Total Runs**: 480 experiments (n=16 concurrent runs per condition)
- **Tasks**: SQL generation, RAG Q&A, JSON summarization
- **Providers**: Ollama (local), IBM watsonx.ai (cloud), OpenAI, Anthropic
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

### 1. 3-Tier Model Classification

Based on output consistency at temperature=0.0:

| Tier | Models | Consistency | Compliance Status |
|------|--------|-------------|-------------------|
| **Tier 1** | 7-8B (Qwen2.5-7B, Granite-3-8B) | **100%** | ✅ Audit-ready |
| **Tier 2** | 40-70B (Llama-3.3-70B, Mistral-Medium) | 56-100% | ⚠️ Task-specific |
| **Tier 3** | 120B+ (GPT-OSS-120B) | **12.5%** | ❌ Non-compliant |

**Interpretation**:
- **Tier 1**: Can be deployed in regulated environments requiring deterministic behavior
- **Tier 2**: Requires careful task-specific validation
- **Tier 3**: Unsuitable for compliance-critical applications despite superior general capabilities

### 2. Task-Specific Results (Temperature=0.0)

| Task Type | Tier 1 (7-8B) | Tier 2 (40-70B) | Tier 3 (120B) |
|-----------|---------------|-----------------|---------------|
| **SQL Generation** | 100% | 100% | 12.5% |
| **Summarization** | 100% | 87.5% | 12.5% |
| **RAG Q&A** | 93.75% | 75.0% | 12.5% |

**Key Insight**: Even for less structured tasks (RAG), Tier 1 models maintain >90% consistency.

### 3. Temperature Sensitivity

RAG task consistency as temperature increases:

| Temperature | Qwen2.5-7B (Tier 1) | Llama-3.3-70B (Tier 2) | GPT-OSS-120B (Tier 3) |
|-------------|---------------------|------------------------|------------------------|
| **T=0.0** | 93.75% | 75.0% | 12.5% |
| **T=0.2** | 56.25% | 43.75% | 6.25% |
| **T=1.0** | 18.75% | 12.5% | 0% |

**Takeaway**: Even small temperature increases (0.0 → 0.2) cause significant drift. For compliance, **T=0.0 is mandatory**.

### 4. Cross-Provider Validation

Testing Tier 1 model consistency across providers:

| Provider Pair | Model | Consistency | Validated |
|---------------|-------|-------------|-----------|
| Ollama ↔ watsonx.ai | Qwen2.5-7B → Granite-3-8B | ≥95% | ✅ |
| Ollama ↔ watsonx.ai | Granite-3-8B → Granite-3-8B | 100% | ✅ |
| Ollama ↔ OpenAI | Qwen2.5-7B → GPT-4 | <50% | ❌ |

**Finding**: Tier 1 models enable **seamless migration** between local (Ollama) and cloud (watsonx.ai) deployments without behavioral changes.

### 5. Regulatory Alignment

Our framework addresses specific regulatory requirements:

| Regulation | Requirement | Framework Solution |
|------------|-------------|--------------------|
| **SR 11-7** | Model validation & ongoing monitoring | Bi-temporal audit trails |
| **ECOA** | Consistent credit decisions | 100% SQL consistency (Tier 1) |
| **FCRA** | Reproducible adverse action rationales | Deterministic RAG retrieval |
| **GDPR Art. 22** | Explainable automated decisions | Citation validation |
| **FSB** | Third-party model risk | Cross-provider validation |
| **CFTC 23.402** | Predictive model documentation | JSONL audit format |

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
- **Solution**: Automated comparison with finance-calibrated tolerance (±5% GAAP)
- **Benefit**: Certify migration safety before production deployment

### 3. Bi-Temporal Audit Trails

JSONL format capturing:

- Input prompt + response hashes (SHA-256)
- Model parameters (temperature, seed, version)
- Compliance metrics (schema validity, citation accuracy)
- Regulatory mappings (SR 11-7, ECOA, FCRA)

---

## Practical Implications

### For Financial Institutions

1. **Vendor Selection**: Prioritize Tier 1 models (7-8B) for compliance-critical tasks
2. **Temperature Policy**: Mandate T=0.0 for all regulated applications
3. **Model Validation**: Use cross-provider validation before production deployment
4. **Audit Trail**: Implement bi-temporal logging per CFTC 23.402 requirements

### For Model Developers

1. **Architecture**: Optimize for determinism, not just accuracy
2. **Testing**: Include multi-run consistency metrics in benchmarks
3. **Documentation**: Report consistency scores alongside performance metrics

### For Regulators

1. **Standards**: Define acceptable consistency thresholds (our research suggests 100% for Tier 1 tasks)
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
@article{khatchadourian2025output,
  title={LLM Output Drift: Financial AI Compliance Framework},
  author={Khatchadourian, Raffi and Franco, Rolando},
  journal={arXiv preprint arXiv:2511.07585},
  year={2025}
}
```

**Paper**: [arXiv:2511.07585](https://arxiv.org/abs/2511.07585) | **DOI**: [10.48550/arXiv.2511.07585](https://doi.org/10.48550/arXiv.2511.07585)

---

## Related Resources

- **Full Paper**: [arXiv:2511.07585](https://arxiv.org/abs/2511.07585)
- **Code Repository**: [GitHub](https://github.com/ibm-client-engineering/output-drift-financial-llms)
- **API Documentation**: [API Reference](api.md)
- **Workshop Labs**: [Lab 0-6](../lab-1/README.md)

---

## Key Takeaways

1. **Size isn't everything**: 7-8B models outperform 120B models for deterministic tasks
2. **Temperature=0.0 is mandatory**: Even T=0.2 causes significant drift
3. **Tier 1 models are audit-ready**: 100% consistency enables regulatory compliance
4. **Cross-provider validation works**: Seamless migration between Ollama and watsonx.ai
5. **Framework is open source**: MIT-licensed, production-ready, extensible

---

**Questions?** See [Troubleshooting Guide](troubleshooting.md) or open an issue on [GitHub](https://github.com/ibm-client-engineering/output-drift-financial-llms/issues).
