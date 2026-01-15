# V3 Real Data Analysis Results

**Date**: December 19, 2025
**Script**: `econometrics/analyze_real_data.py`
**Data Source**: `results/aggregate.csv` (V2 experiment data)

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Configurations | 74 |
| Temperature 0.0 | 51 |
| Temperature 0.2 | 23 |
| Unique Models | 12 |
| Providers | ollama, anthropic, gemini, watsonx |
| Tasks | rag, sql, summary |

---

## Model Tier Analysis

### Tier 1 (7-20B Parameters)

| Metric | Value |
|--------|-------|
| Models | qwen2.5:7b-instruct, gpt-oss:20b, deepseek-r1:8b, granite-3-8b-instruct |
| Configurations | 36 |
| Mean Determinism | **100.0%** (std: 0.0) |
| Mean Faithfulness | **100.0%** |
| Validation Scaling | **1.0x** (baseline) |

**Task Breakdown:**
- RAG: 100.0% determinism
- SQL: 100.0% determinism
- Summary: 100.0% determinism

**Key Finding**: Tier 1 models achieve perfect determinism across ALL tasks at T=0.

---

### Frontier (Claude, Gemini)

| Metric | Value |
|--------|-------|
| Models | claude-opus-4-5-20251101, gemini-2.5-pro |
| Configurations | 12 |
| Mean Determinism | 88.5% (std: 18.8) |
| Mean Faithfulness | 100.0% |
| Validation Scaling | 1.34x |

**Task Breakdown:**
- RAG: 68.8% determinism
- SQL: 100.0% determinism
- Summary: 96.9% determinism

**Key Finding**: Frontier models maintain 100% faithfulness but show task-dependent determinism. RAG is the weak point.

---

### Tier 2 (40-70B Parameters)

| Metric | Value |
|--------|-------|
| Models | llama-3-3-70b-instruct, granite-3-8b-instruct (via watsonx) |
| Configurations | 20 |
| Mean Determinism | 73.4% (std: 27.2) |
| Mean Faithfulness | 75.0% |
| Validation Scaling | 1.8x |

**Task Breakdown:**
- RAG: 66.6% determinism
- SQL: 87.5% determinism
- Summary: 79.7% determinism

**Key Finding**: Significant variance. SQL remains most stable. 1.8x more validation samples needed.

---

### Tier 3 (120B+ Parameters)

| Metric | Value |
|--------|-------|
| Models | openai/gpt-oss-120b |
| Configurations | 6 |
| Mean Determinism | **9.7%** (std: 15.7) |
| Mean Faithfulness | 71.9% |
| Validation Scaling | **3.71x** |

**Task Breakdown:**
- RAG: 2.1% determinism
- SQL: 22.9% determinism
- Summary: 4.2% determinism

**Key Finding**: Extremely low determinism across all tasks. Requires 3.7x validation samples. NOT recommended for audit-critical tasks.

---

## Task-Structure Effect

Determinism varies significantly by task type, confirming the **task-structure hypothesis**:

| Task | Temp | Det. Mean | Det. Std | n | Faith. Mean | Drift Mean |
|------|------|-----------|----------|---|-------------|------------|
| **SQL** | 0.0 | **92.71%** | 23.64 | 20 | 97.50% | 0.00 |
| SQL | 0.2 | 91.67% | 22.05 | 7 | 99.70% | 0.00 |
| **Summary** | 0.0 | 90.07% | 26.59 | 17 | 97.06% | 0.00 |
| Summary | 0.2 | 83.33% | 35.76 | 7 | 100.00% | 0.00 |
| **RAG** | 0.0 | 75.89% | 29.61 | 14 | 79.91% | 0.02 |
| RAG | 0.2 | 57.29% | 35.77 | 9 | 68.29% | 0.07 |

**Interpretation**:
1. **SQL** (structured output): Highest determinism - JSON/SQL schemas constrain variance
2. **Summary** (semi-structured): High determinism - fixed-length outputs help
3. **RAG** (open-ended): Lowest determinism - retrieval introduces variance

---

## Faithfulness-Determinism Relationship

### Correlation Analysis (T=0.0 only)

| Metric | Value |
|--------|-------|
| Pearson Correlation | **0.453** |
| Interpretation | **Positive** (not a trade-off!) |

### Quadrant Distribution

| Quadrant | Count | % |
|----------|-------|---|
| High Det + High Faith | 38 | 74.5% |
| High Det + Low Faith | 1 | 2.0% |
| Low Det + High Faith | 7 | 13.7% |
| Low Det + Low Faith | 5 | 9.8% |

**Key Finding**: Models that are more deterministic are ALSO more faithful. This contradicts the common assumption of a trade-off. 74.5% of configs achieve both high determinism AND high faithfulness.

---

## Econometric Research Recommendations

### For Maximum Label Stability
Use **Tier 1 (7-20B)** models at T=0.0:
- 100% determinism
- 100% faithfulness
- 1.0x validation scaling (baseline)

### For Maximum Accuracy
Use **Tier 1 (7-20B)** models (same recommendation):
- Perfect determinism AND faithfulness
- No trade-off required

### Validation Sample Scaling

| Tier | Drift Rate | Validation Scaling | Sample Size for 100 baseline |
|------|------------|-------------------|------------------------------|
| Tier 1 | 0.0% | 1.0x | 100 |
| Frontier | 11.5% | 1.34x | 134 |
| Tier 2 | 26.6% | 1.8x | 180 |
| Tier 3 | 90.3% | 3.71x | 371 |

### Task-Specific Recommendations

1. **Tier 2 (40-70B)**: Use for SQL/structured tasks (88% det) but **avoid RAG** (67% det)
2. **Frontier**: Use for SQL/structured tasks (100% det) but **avoid RAG** (69% det)
3. **Tier 3 (120B+)**: Use for SQL/structured tasks (23% det) but **avoid RAG** (2% det)

---

## Reproduction Commands

```bash
# Run the analysis
python econometrics/analyze_real_data.py

# Expected output summary:
# - Loaded 74 configurations across 12 models
# - 4 providers: ollama, anthropic, gemini, watsonx
# - 3 tasks: rag, sql, summary
# - Correlation: 0.453 (positive)
```

---

## Paper Implications

These findings support the following claims for the ICLR 2026 paper:

1. **Smaller models are more reliable** for audit-critical financial tasks
2. **Task structure matters**: SQL > Summary > RAG for determinism
3. **No determinism-faithfulness trade-off**: Positive correlation challenges conventional wisdom
4. **Validation scaling is model-dependent**: Tier 3 requires 3.7x more samples
5. **Temperature 0.0 is not enough**: Tier 3 still shows 90% drift even at T=0

---

## Next Steps

1. [x] Run analyze_real_data.py - DONE
2. [ ] Draft LaTeX paper with these tables
3. [ ] Design 3 benchmark tasks
4. [ ] Create stress-test harness for public release
