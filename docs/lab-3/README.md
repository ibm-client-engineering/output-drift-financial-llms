# Lab 3: Running Your First Experiment

## Overview

In this lab, you'll run a complete drift evaluation experiment, like the
historical workshop study. You'll vary concurrency, temperature, and task type
and observe how repeatability changes in the resulting runs.

**Duration**: ~30 minutes

## Learning Objectives

By the end of this lab, you will:

- Run experiments with varying concurrency (1, 4, 16 runs)
- Compare drift at temperature 0.0 vs 0.2
- Understand how task types affect consistency
- Analyze JSONL replay records
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
  --providers ollama \
  --models qwen2.5:7b-instruct \
  --temperatures 0.0 \
  --concurrency 1 \
  --tasks sql \
  --repeats 1 \
  --output traces/lab3_step1
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

Replay record: traces/lab3_step1/trace_*.jsonl
✅ Single-run baseline complete!
```

**Analysis**: With n=1, we can't measure drift yet. We need multiple runs.

## Step 2: Low Concurrency Test (n=4)

Now let's run 4 concurrent queries:

```bash
python run_evaluation.py \
  --providers ollama \
  --models qwen2.5:7b-instruct \
  --temperatures 0.0 \
  --concurrency 4 \
  --tasks sql \
  --repeats 4 \
  --output traces/lab3_step2
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

✅ Exact output agreement at n=4!
```

!!! success "Tier 1 Performance"
    The tested 7-20B configurations retained 100% observed agreement under the
    exercise's concurrent requests. Re-run the test for the intended serving
    stack before drawing an operational conclusion.

## Step 3: Paper-Standard Test (n=16)

Now run the same configuration used in the paper (n=16):

```bash
python run_evaluation.py \
  --providers ollama \
  --models qwen2.5:7b-instruct \
  --temperatures 0.0 \
  --concurrency 16 \
  --tasks sql \
  --repeats 16 \
  --output traces/lab3_step3
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

✅ Exact output agreement at n=16!
```

**Bounded result**: Qwen2.5-7B reached **100% consistency** across these 16
runs, which meets the historical workshop's Tier 1 definition. Re-run the
exact intended configuration before relying on that label elsewhere.

## Step 4: Temperature Sensitivity Test

Now let's test what happens when we increase temperature to 0.2:

```bash
python run_evaluation.py \
  --providers ollama \
  --models qwen2.5:7b-instruct \
  --temperatures 0.2 \
  --concurrency 16 \
  --tasks sql \
  --repeats 16 \
  --output traces/lab3_step4
```

**Expected output (SQL task):**

```
Running 16 concurrent queries...
████████████████████████████████ 16/16 [00:12]

Results:
  Consistency: 100.0% (16/16 identical)
  Mean Drift: 0.000
  Temperature: 0.2

✅ SQL outputs matched exactly in this T=0.2 run!
```

!!! info "Structured Task Resilience"
    The SQL outputs reached 100% exact agreement at T=0.2 in the workshop
    result. The structured format may constrain variation, but this single
    comparison does not establish a causal explanation or guarantee future
    replay.

## Step 5: RAG Task Comparison

Now let's test a RAG task, which our paper shows is more susceptible to drift:

```bash
python run_evaluation.py \
  --providers ollama \
  --models qwen2.5:7b-instruct \
  --temperatures 0.0 \
  --concurrency 16 \
  --tasks rag \
  --repeats 16 \
  --output traces/lab3_step5_t0
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

✅ The shown outputs differ in form while retaining the same displayed value and citation.
```

!!! note "RAG vs SQL"
    RAG tasks show slightly lower consistency (93.75% vs 100%) due to:

    - Broader output space (natural language)
    - Retrieval context variations
    - Formatting flexibility

Now test RAG at T=0.2:

```bash
python run_evaluation.py \
  --providers ollama \
  --models qwen2.5:7b-instruct \
  --temperatures 0.2 \
  --concurrency 16 \
  --tasks rag \
  --repeats 16 \
  --output traces/lab3_step5_t02
```

**Expected output (from paper findings):**

```
Results:
  Consistency: 56.25% (9/16 identical)
  Mean Drift: 0.081
  Token-Set Drift Range: 0.000 - 0.375

⚠️ Substantial drift at T=0.2 for RAG tasks!
```

**Historical result reproduced**: the tested RAG configuration at T=0.2
shows **56.25% consistency**. This result calls for investigation; it does not
by itself decide whether a workflow is compliant or suitable for use.

## Step 6: Multi-Task Evaluation

Run all three task types in sequence:

```bash
# Run all three tasks at once
python run_evaluation.py \
  --providers ollama \
  --models qwen2.5:7b-instruct \
  --temperatures 0.0 \
  --concurrency 16 \
  --tasks rag,summary,sql \
  --repeats 16 \
  --output traces/lab3_multi
```

**Summary script** to compare results:

Create `analyze_lab3.py`:

```python
from collections import Counter, defaultdict
import json
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

tasks = ["sql", "summary", "rag"]
results = []

trace_files = list(Path("traces/lab3_multi").glob("trace_*.jsonl"))
if len(trace_files) != 1:
    raise RuntimeError(f"Expected one trace file, found {len(trace_files)}")

with trace_files[0].open() as f:
    all_records = [json.loads(line) for line in f]

for task in tasks:
    data = [record for record in all_records if record["task"] == task]
    groups = defaultdict(list)
    for record in data:
        groups[record["prompt_id"]].append(record)

    modal_count = sum(
        Counter(record["output"] for record in group).most_common(1)[0][1]
        for group in groups.values()
    )
    consistency_pct = modal_count / len(data) * 100

    distances = []
    for group in groups.values():
        reference = group[0]["output"]
        distances.extend(
            Levenshtein.normalized_distance(reference, record["output"])
            for record in group
        )

    results.append({
        "Task": task.upper(),
        "Runs": len(data),
        "Modal exact agreement": f"{consistency_pct:.1f}%",
        "Mean normalized distance": f"{sum(distances) / len(distances):.3f}",
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

The script prints one row per task. Counts reflect all built-in prompt IDs, so
they may be larger than the one-prompt historical examples used elsewhere in
this lab:

```
📊 Multi-Task Evaluation Results (T=0.0, n=16)
============================================================
      Task  Runs Modal exact agreement Mean normalized distance
       SQL    ...                  ...%                      ...
   SUMMARY    ...                  ...%                      ...
       RAG    ...                  ...%                      ...
```

## Understanding the Results

!!! note "Interpretation boundary"
    These thresholds summarize replay variation in the exercise. They are
    investigation cues, not compliance or safety thresholds.

### Consistency Metric

**Formula**: within each prompt's replay group, count the modal exact output;
then divide the summed modal counts by the total runs.

- **100%**: All responses identical (byte-for-byte)
- **93.75%**: 15/16 responses identical, 1 syntactic variant
- **<90%**: Significant drift; investigate before relying on exact replay

### Mean String-Distance Metric

**Formula in `run_evaluation.py`**: normalized Levenshtein distance from each
replay group's first output.

- **0.000**: No measured drift in this run
- **0.012**: Minor syntactic variation
- **>0.05**: Larger token-set variation
- **>0.1**: Substantial token-set variation; inspect the outputs directly

### Paper Findings Reproduced

| Task | Expected (Paper) | Your Results | Match? |
|------|-----------------|--------------|--------|
| SQL (T=0.0) | 100% | 100% | ✅ |
| Summarize (T=0.0) | 100% | 100% | ✅ |
| RAG (T=0.0) | 93.75% | ~94% | ✅ |

## Analyzing Replay Records

Replay records are stored as JSONL (JSON Lines), one JSON object per line.

**View a specific run:**

```bash
# Pretty-print the 5th run from the isolated Lab 3 directory
sed -n '5p' traces/lab3_multi/trace_*.jsonl | python -m json.tool
```

**Example entry:**

```json
{
  "ts": 1762525425123,
  "ts_end": 1762525426368,
  "task": "sql",
  "model": "qwen2.5:7b-instruct",
  "provider": "ollama",
  "temp": 0.0,
  "conc": 16,
  "prompt_id": "s1",
  "prompt": "Compute total \"amount\" across all transactions.",
  "output": "SELECT SUM(amount) FROM transactions;",
  "decision_ok": true,
  "latency": 1.245
}
```

**Key fields for review:**

- `task` and `prompt_id`: identify the replay group
- `provider`, `model`, `temp`, and `conc`: record the captured configuration
- `prompt` and `output`: preserve the request text and observed response
- `decision_ok`, `schema_violation`, or `citations`: task-specific fields when available
- `ts`, `ts_end`, and `latency`: timing fields for the recorded call

## Comparing Replay Records

Compare two runs to find differences:

```python
import hashlib
import json
from pathlib import Path

# Load two runs
trace_file = next(Path("traces/lab3_multi").glob("trace_*.jsonl"))
with trace_file.open() as f:
    lines = f.readlines()

run1 = json.loads(lines[0])
run2 = json.loads(lines[1])

hash1 = hashlib.sha256(run1["output"].encode()).hexdigest()
hash2 = hashlib.sha256(run2["output"].encode()).hexdigest()
print("Run 1 output hash:", hash1)
print("Run 2 output hash:", hash2)
print("Identical?", hash1 == hash2)

if run1["output"] != run2["output"]:
    print("\nResponse Diff:")
    print("Run 1:", run1["output"])
    print("Run 2:", run2["output"])
else:
    print("\n✅ Responses are identical!")
```

## Advanced: Historical Experiment Subset

To run the three locally named configurations shown below across the listed
conditions, budget for up to **1,728 calls** (3 models × 2 temperatures × 3
concurrency settings × 3 tasks × 2 built-in prompts × 16 repeats):

```bash
python run_evaluation.py \
  --providers ollama \
  --models qwen2.5:7b-instruct,granite-3-8b,llama-3.3-70b \
  --temperatures 0.0,0.2 \
  --concurrency 1,4,16 \
  --tasks rag,summary,sql \
  --repeats 16 \
  --output traces/lab3_historical_subset
```

!!! warning "Resource Intensive"
    This subset requires:
    - The three listed models available
    - Capacity and time for up to 1,728 calls
    - Sufficient local storage for the resulting traces

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

```bash
# The historical runner has no rate-limit flag. Lower concurrency to reduce
# simultaneous requests, and use the provider SDK/account retry controls.
python run_evaluation.py \
  --providers anthropic \
  --models YOUR_MODEL_ID \
  --concurrency 1 \
  --tasks sql \
  --repeats 4 \
  --output traces/rate_limit_check
```

### Out of Memory

For large concurrency (n=16):

```bash
# Reduce simultaneous requests
python run_evaluation.py \
  --providers ollama \
  --models qwen2.5:7b-instruct \
  --concurrency 4 \
  --tasks sql \
  --repeats 16 \
  --output traces/lower_concurrency
```

## Key Takeaways

1. **The tested 7-20B configurations** reached 100% consistency at T=0.0 in this exercise
2. **Concurrency did not change measured consistency** for those runs (n=1, 4, or 16)
3. **Task structure mattered in these runs**: SQL/summarization had higher exact-output agreement than RAG
4. **Temperature sensitivity was visible**: the tested RAG condition had lower agreement at T=0.2
5. **Replay records** support reproduction and review within their captured scope

## Quiz: Test Your Understanding

??? question "Why does SQL maintain 100% consistency even at T=0.2?"
    **Answer**: The tested SQL outputs reached 100% exact agreement at T=0.2.
    A structured format can constrain the output space, but the exercise does
    not prove that structure caused the result or that it will recur.

??? question "What consistency % did RAG tasks achieve at T=0.2 in the paper?"
    **Answer**: 56.25% (9/16 runs identical), showing substantial variation
    that should be investigated before relying on exact replay.

??? question "How can you compare outputs when a trace has no stored output hash?"
    **Answer**: Hash the captured `output` value during analysis, as shown
    above. Keep the raw output if later review needs more than equality.

## Next Steps

Now that you've run experiments and understand the methodology:

1. **Proceed to [Lab 4: Analyzing Drift Metrics](../lab-4/README.md)** to visualize and interpret results
2. Explore different prompts in `prompts/templates.json`
3. Try modifying temperature and concurrency parameters

---

!!! success "Lab 3 Complete!"
    You've run drift evaluations and compared your results with the historical
    workshop examples. Ready to analyze the data? Move on to
    [Lab 4: Analyzing Drift Metrics](../lab-4/README.md)!
