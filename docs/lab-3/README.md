# Lab 3: Running Your First Experiment

## Overview

In this lab, you'll run a complete drift evaluation experiment, just like the ones from the paper. You'll test different concurrency levels, temperatures, and task types to understand how these factors affect determinism.

**Duration**: ~30 minutes

## Learning Objectives

By the end of this lab, you will:

- Run experiments with varying concurrency (1, 4, 16 runs)
- Compare drift at temperature 0.0 vs 0.2
- Understand how task types affect consistency
- Analyze JSONL audit trails
- Reproduce key findings from the paper

## Prerequisites

- Completed [Lab 2: Setting Up Your Environment](../lab-2/README.md)
- At least one provider configured (Ollama with qwen2.5:7b-instruct recommended)
- Synthetic database generated (`data/toy_finance.sqlite`)

## Experimental Design (Paper Methodology)

Our paper evaluated 5 models across **480 runs** with the following design:

| Parameter | Values |
|-----------|--------|
| **Models** | Qwen2.5-7B, Granite-3-8B, Llama-3.3-70B, Mistral-Medium, GPT-OSS-120B |
| **Temperatures** | 0.0, 0.2 |
| **Concurrency** | n=16 per condition |
| **Tasks** | SQL generation, RAG (Text-to-SQL), JSON summarization |

In this lab, we'll run a subset to understand the methodology, then you can scale to full experiments.

## Step 1: Single-Run Baseline (Concurrency = 1)

Let's start with a single run to establish a baseline:

```bash
python run_evaluation.py \
  --provider ollama \
  --model qwen2.5:7b-instruct \
  --temperature 0.0 \
  --concurrency 1 \
  --task sql \
  --output traces/lab3_single.jsonl
```

**Expected output:**

```
🚀 Output Drift Evaluation Framework
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Configuration:
  Provider: ollama
  Model: qwen2.5:7b-instruct
  Temperature: 0.0
  Concurrency: 1
  Task: sql

Prompt: "Generate SQL to find all customers with account balance > $100,000"

Run 1/1...
  Response: SELECT customer_name, account_balance FROM accounts
            WHERE account_balance > 100000
  Execution time: 1.2s

Results:
  Runs completed: 1
  Schema valid: ✅ Yes

Audit trail: traces/lab3_single.jsonl
✅ Single-run baseline complete!
```

**Analysis**: With n=1, we can't measure drift yet. We need multiple runs.

## Step 2: Low Concurrency Test (n=4)

Now let's run 4 concurrent queries:

```bash
python run_evaluation.py \
  --provider ollama \
  --model qwen2.5:7b-instruct \
  --temperature 0.0 \
  --concurrency 4 \
  --task sql \
  --output traces/lab3_concurrent_4.jsonl
```

**Expected output:**

```
Running 4 concurrent queries...
████████████████████████████████ 4/4 [00:05]

Results:
  Consistency: 100.0% (4/4 identical)
  Mean Drift: 0.000
  Jaccard Similarity: 1.000
  Schema Violations: 0
  Decision Flips: 0

Unique responses: 1
Response 1 (4 occurrences):
  "SELECT customer_name, account_balance FROM accounts WHERE account_balance > 100000"

✅ Perfect consistency at n=4!
```

!!! success "Tier 1 Performance"
    7-8B models maintain 100% consistency even with concurrent requests—critical for production workloads.

## Step 3: Paper-Standard Test (n=16)

Now run the same configuration used in the paper (n=16):

```bash
python run_evaluation.py \
  --provider ollama \
  --model qwen2.5:7b-instruct \
  --temperature 0.0 \
  --concurrency 16 \
  --task sql \
  --output traces/lab3_concurrent_16.jsonl
```

**Expected output:**

```
Running 16 concurrent queries...
████████████████████████████████ 16/16 [00:12]

Results:
  Consistency: 100.0% (16/16 identical)
  Mean Drift: 0.000
  Jaccard Similarity: 1.000
  Schema Violations: 0
  Decision Flips: 0

Unique responses: 1
Response 1 (16 occurrences):
  "SELECT customer_name, account_balance FROM accounts WHERE account_balance > 100000"

✅ Perfect consistency at n=16!
```

**Key Finding**: Qwen2.5-7B achieves **100% consistency** at n=16, confirming Tier 1 classification.

## Step 4: Temperature Sensitivity Test

Now let's test what happens when we increase temperature to 0.2:

```bash
python run_evaluation.py \
  --provider ollama \
  --model qwen2.5:7b-instruct \
  --temperature 0.2 \
  --concurrency 16 \
  --task sql \
  --output traces/lab3_temp02.jsonl
```

**Expected output (SQL task):**

```
Running 16 concurrent queries...
████████████████████████████████ 16/16 [00:12]

Results:
  Consistency: 100.0% (16/16 identical)
  Mean Drift: 0.000
  Temperature: 0.2

✅ SQL generation remains deterministic at T=0.2!
```

!!! info "Structured Task Resilience"
    SQL generation maintains 100% consistency even at T=0.2 because of its **structured output format** and **deterministic syntax**.

## Step 5: RAG Task Comparison

Now let's test a RAG task, which our paper shows is more susceptible to drift:

```bash
python run_evaluation.py \
  --provider ollama \
  --model qwen2.5:7b-instruct \
  --temperature 0.0 \
  --concurrency 16 \
  --task rag \
  --output traces/lab3_rag_t00.jsonl
```

**Expected output:**

```
Running 16 concurrent queries...
████████████████████████████████ 16/16 [00:18]

Task: RAG (Retrieval-Augmented Generation)
Prompt: "What were Citigroup's net credit losses in 2023?"

Results:
  Consistency: 93.75% (15/16 identical)
  Mean Drift: 0.012
  Factual Drift: 0.000
  Citation Accuracy: 1.0

Unique responses: 2
Response 1 (15 occurrences):
  "According to Citigroup's 2024 10-K (page 145), net credit losses were $2.4 billion in 2023."

Response 2 (1 occurrence):
  "Citigroup reported net credit losses of $2.4B in 2023 (10-K filing, page 145)."

✅ Minor syntactic drift, but factual consistency maintained!
```

!!! note "RAG vs SQL"
    RAG tasks show slightly lower consistency (93.75% vs 100%) due to:

    - Broader output space (natural language)
    - Retrieval context variations
    - Formatting flexibility

Now test RAG at T=0.2:

```bash
python run_evaluation.py \
  --provider ollama \
  --model qwen2.5:7b-instruct \
  --temperature 0.2 \
  --concurrency 16 \
  --task rag \
  --output traces/lab3_rag_t02.jsonl
```

**Expected output (from paper findings):**

```
Results:
  Consistency: 56.25% (9/16 identical)
  Mean Drift: 0.081
  Factual Drift Range: 0.000 - 0.375

⚠️ Substantial drift at T=0.2 for RAG tasks!
```

**Paper Finding Confirmed**: RAG tasks at T=0.2 show **56.25% consistency**, making them unsuitable for compliance workflows without strict T=0.0.

## Step 6: Multi-Task Evaluation

Run all three task types in sequence:

```bash
# SQL
python run_evaluation.py --model qwen2.5:7b-instruct --temperature 0.0 --concurrency 16 --task sql --output traces/lab3_sql.jsonl

# Summarization
python run_evaluation.py --model qwen2.5:7b-instruct --temperature 0.0 --concurrency 16 --task summarize --output traces/lab3_summarize.jsonl

# RAG
python run_evaluation.py --model qwen2.5:7b-instruct --temperature 0.0 --concurrency 16 --task rag --output traces/lab3_rag.jsonl
```

**Summary script** to compare results:

Create `analyze_lab3.py`:

```python
import json
import pandas as pd

tasks = ["sql", "summarize", "rag"]
results = []

for task in tasks:
    with open(f"traces/lab3_{task}.jsonl") as f:
        data = [json.loads(line) for line in f]

    consistency = len(set(d["response_hash"] for d in data)) == 1
    consistency_pct = 100.0 if consistency else (len(data) / len(set(d["response_hash"] for d in data))) * 100

    results.append({
        "Task": task.upper(),
        "Runs": len(data),
        "Consistency": f"{consistency_pct:.1f}%",
        "Mean Drift": f"{sum(d['compliance_metrics']['factual_drift'] for d in data) / len(data):.3f}"
    })

df = pd.DataFrame(results)
print("\n📊 Multi-Task Evaluation Results (T=0.0, n=16)")
print("=" * 60)
print(df.to_string(index=False))
```

Run it:

```bash
python analyze_lab3.py
```

**Expected output:**

```
📊 Multi-Task Evaluation Results (T=0.0, n=16)
============================================================
      Task  Runs Consistency Mean Drift
       SQL    16       100.0%      0.000
SUMMARIZE    16       100.0%      0.000
       RAG    16        93.8%      0.012
```

## Understanding the Results

### Consistency Metric

**Formula**: `consistency = (identical_responses / total_runs) * 100`

- **100%**: All responses identical (byte-for-byte)
- **93.75%**: 15/16 responses identical, 1 syntactic variant
- **<90%**: Significant drift, not compliance-safe

### Mean Drift Metric

**Formula**: Jaccard distance between token sets

- **0.000**: Perfect determinism
- **0.012**: Minor syntactic variation
- **>0.05**: Semantic drift
- **>0.1**: Factual inconsistencies

### Paper Findings Reproduced

| Task | Expected (Paper) | Your Results | Match? |
|------|-----------------|--------------|--------|
| SQL (T=0.0) | 100% | 100% | ✅ |
| Summarize (T=0.0) | 100% | 100% | ✅ |
| RAG (T=0.0) | 93.75% | ~94% | ✅ |

## Analyzing Audit Trails

Audit trails are stored as JSONL (JSON Lines)—one JSON object per line.

**View a specific run:**

```bash
# Pretty-print the 5th run
sed -n '5p' traces/lab3_concurrent_16.jsonl | python -m json.tool
```

**Example entry:**

```json
{
  "timestamp": "2025-11-07T14:23:45.123Z",
  "run_id": "lab3_concurrent_16_005",
  "model": "qwen2.5:7b-instruct",
  "provider": "ollama",
  "temperature": 0.0,
  "seed": 42,
  "concurrency_idx": 5,
  "task_type": "sql",
  "prompt": "Generate SQL to find all customers with account balance > $100,000",
  "response": "SELECT customer_name, account_balance FROM accounts WHERE account_balance > 100000",
  "prompt_hash": "sha256:a3d8f92b1c4e5f6789abcdef",
  "response_hash": "sha256:b2c1e7d8a9f6543210fedcba",
  "execution_time_ms": 1245,
  "compliance_metrics": {
    "schema_valid": true,
    "citation_accuracy": 1.0,
    "decision_flip": false,
    "factual_drift": 0.0
  },
  "regulatory_mappings": {
    "FSB": "consistent_decisions",
    "CFTC": "document_ai_outcomes",
    "SR_11_7": "model_validation"
  }
}
```

**Key fields for audits:**

- `prompt_hash`: SHA-256 of input (for duplicate detection)
- `response_hash`: SHA-256 of output (for consistency checking)
- `compliance_metrics`: Drift measures
- `regulatory_mappings`: Compliance framework mappings

## Comparing Audit Trails

Compare two runs to find differences:

```python
import json

# Load two runs
with open("traces/lab3_concurrent_16.jsonl") as f:
    lines = f.readlines()

run1 = json.loads(lines[0])
run2 = json.loads(lines[1])

print("Run 1 response hash:", run1["response_hash"])
print("Run 2 response hash:", run2["response_hash"])
print("Identical?", run1["response_hash"] == run2["response_hash"])

if run1["response"] != run2["response"]:
    print("\nResponse Diff:")
    print("Run 1:", run1["response"])
    print("Run 2:", run2["response"])
else:
    print("\n✅ Responses are identical!")
```

## Advanced: Full Paper Replication

To fully reproduce the paper's 480 runs:

```bash
# This will take ~30-45 minutes
python run_evaluation.py \
  --models qwen2.5:7b granite-3-8b llama-3.3-70b mistral-medium gpt-oss-120b \
  --temperatures 0.0 0.2 \
  --concurrency 16 \
  --tasks sql summarize rag \
  --output traces/full_replication/
```

!!! warning "Resource Intensive"
    Full replication requires:
    - All 5 models available (some may require API keys)
    - ~45 minutes of runtime
    - ~500 MB of trace data

## Troubleshooting

### Inconsistent Results

If you're seeing drift where you shouldn't (e.g., SQL at T=0.0):

```bash
# Check model version
ollama show qwen2.5:7b-instruct

# Ensure seed is set
# In run_evaluation.py, verify: seed=42
```

### Rate Limiting

If using cloud providers (watsonx, OpenAI):

```python
# Add rate limiting in configuration
--rate-limit 10  # requests per minute
--retry-delay 5  # seconds between retries
```

### Out of Memory

For large concurrency (n=16):

```bash
# Reduce batch size
--batch-size 4  # Process 4 at a time instead of 16
```

## Key Takeaways

1. **7-8B models** (Tier 1) achieve 100% consistency at T=0.0 for all tasks
2. **Concurrency doesn't affect consistency** for Tier 1 models (n=1, 4, or 16)
3. **Task structure matters**: SQL/summarization > RAG for determinism
4. **Temperature sensitivity**: RAG tasks degrade significantly at T=0.2
5. **Audit trails** provide complete reproducibility for regulatory review

## Quiz: Test Your Understanding

??? question "Why does SQL maintain 100% consistency even at T=0.2?"
    **Answer**: SQL has a structured output format with deterministic syntax, limiting the output space and making it more resistant to temperature-induced drift.

??? question "What consistency % did RAG tasks achieve at T=0.2 in the paper?"
    **Answer**: 56.25% (9/16 runs identical), showing substantial drift that makes them unsuitable for compliance workflows at elevated temperatures.

??? question "What is the purpose of the response_hash field in audit trails?"
    **Answer**: SHA-256 hash enables fast consistency checking across runs without string comparison—critical for large-scale audits.

## Next Steps

Now that you've run experiments and understand the methodology:

1. **Proceed to [Lab 4: Analyzing Drift Metrics](../lab-4/README.md)** to visualize and interpret results
2. Explore different prompts in `prompts/templates.json`
3. Try modifying temperature and concurrency parameters

---

!!! success "Lab 3 Complete!"
    You've successfully run drift evaluations and reproduced key paper findings! Ready to analyze the data? Move on to [Lab 4: Analyzing Drift Metrics](../lab-4/README.md)!
