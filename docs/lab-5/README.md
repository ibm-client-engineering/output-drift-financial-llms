# Lab 5: Cross-Provider Testing

## Overview

In this lab, you'll compare outputs from local (Ollama) and cloud (IBM
watsonx.ai) configurations using the framework's `CrossProviderValidator`.
The comparison measures agreement in the captured examples; it does not
establish provider-independent reliability.

**Duration**: ~30 minutes

## Learning Objectives

By the end of this lab, you will:

- Use the `CrossProviderValidator` from your framework
- Compare outputs between Ollama and watsonx.ai
- Use an illustrative numeric tolerance (±5%)
- Measure cross-provider consistency for follow-up review
- Prioritize validation work when providers differ

## Prerequisites

- Completed [Lab 4: Analyzing Drift Metrics](../lab-4/README.md)
- **At least two providers configured**: Ollama + one cloud provider (watsonx.ai recommended)
- API keys in `.env` file

## Why Cross-Provider Validation Matters

Financial institutions often need to:

- **Migrate** between providers without changing behavior
- **Redundancy** with failover to backup providers
- **Vendor independence** to avoid lock-in
- **Governance and review** when replay evidence changes across environments

!!! warning "The Risk"
    A model that works locally but behaves differently in production (cloud)
    creates inconsistent replay evidence and may undermine downstream controls.
    The consistency test does not itself determine compliance.

## Step 1: Review CrossProviderValidator Code

Open `harness/cross_provider_validation.py` to see how it works:

```bash
cat harness/cross_provider_validation.py | head -50
```

**Key features** (from the code):
- Normalized edit distance for text comparison
- Configurable numeric tolerance (±5% in this exercise)
- Task-specific validation rules
- Audit trail generation

## Step 2: Test Ollama vs watsonx.ai

Create `test_cross_provider.py`:

```python
#!/usr/bin/env python3
"""
Cross-provider validation test: Ollama (local) vs watsonx.ai (cloud)
"""
import os
from openai import OpenAI
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from dotenv import load_dotenv

load_dotenv()

# Test prompt (SQL generation)
prompt = "Generate SQL to find all customers with account balance > $100,000"

print("🔄 Cross-Provider Validation Test")
print("=" * 60)
print(f"Prompt: {prompt}\n")

# Provider 1: Ollama (local)
print("📍 Provider 1: Ollama (qwen2.5:7b-instruct)")
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
ollama_response = ollama_client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
    seed=42
)
ollama_output = ollama_response.choices[0].message.content
print(f"Output: {ollama_output}\n")

# Provider 2: IBM watsonx.ai (cloud)
print("☁️  Provider 2: watsonx.ai (granite-3-8b-instruct)")
watsonx_model = ModelInference(
    model_id="ibm/granite-3-8b-instruct",
    api_key=os.getenv("WATSONX_API_KEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
)
watsonx_params = {
    GenParams.TEMPERATURE: 0.0,
    GenParams.MAX_NEW_TOKENS: 200,
    GenParams.RANDOM_SEED: 42
}
watsonx_response = watsonx_model.generate_text(prompt=prompt, params=watsonx_params)
watsonx_output = watsonx_response
print(f"Output: {watsonx_output}\n")

# Compare outputs
print("=" * 60)
print("🔍 Comparison:")
print(f"  Ollama length: {len(ollama_output)} chars")
print(f"  watsonx length: {len(watsonx_output)} chars")
print(f"  Exact match: {ollama_output == watsonx_output}")

# Calculate similarity (Levenshtein distance)
from rapidfuzz.distance import Levenshtein
distance = Levenshtein.normalized_distance(ollama_output, watsonx_output)
similarity = 1.0 - distance
print(f"  Similarity: {similarity:.1%}")

if similarity >= 0.95:
    print("\n✅ Observed similarity met the exercise's 95% review line")
else:
    print(f"\n⚠️  Cross-provider drift detected: {similarity:.1%}")
```

Run it:

```bash
python test_cross_provider.py
```

**Expected output (both Tier 1 models):**

```
🔄 Cross-Provider Validation Test
============================================================
Prompt: Generate SQL to find all customers with account balance > $100,000

📍 Provider 1: Ollama (qwen2.5:7b-instruct)
Output: SELECT customer_name, account_balance FROM accounts WHERE account_balance > 100000

☁️  Provider 2: watsonx.ai (granite-3-8b-instruct)
Output: SELECT customer_name, account_balance FROM accounts WHERE account_balance > 100000

============================================================
🔍 Comparison:
  Ollama length: 82 chars
  watsonx length: 82 chars
  Exact match: True
  Similarity: 100.0%

✅ Observed similarity met the exercise's 95% review line
```

!!! success "Tier 1 Cross-Provider Consistency"
    Granite-3-8B (watsonx) and Qwen2.5-7B (Ollama) produced **identical
    outputs in this example**. A migration decision still requires replay of
    the intended tasks plus correctness, path, cost, latency, and control
    review.

## Step 3: Use the Framework's CrossProviderValidator

Now use the built-in validator from `harness/cross_provider_validation.py`:

Create `run_cross_provider_validation.py`:

```python
#!/usr/bin/env python3
"""
Use framework's CrossProviderValidator for automated testing.

Note: The validator works with pre-collected outputs. You first run
experiments on each provider, then validate the outputs for consistency.
"""
from harness.cross_provider_validation import CrossProviderValidator

# Initialize the validator with an exercise-specific tolerance
validator = CrossProviderValidator(
    providers=["ollama", "watsonx"],
    tolerance_pct=5.0  # illustrative; set from the approved task contract
)

# Assume you've already collected outputs from each provider
# (e.g., from running run_evaluation.py with --providers ollama and --providers watsonx)
sql_outputs = {
    "ollama": "SELECT customer_id, name, balance FROM accounts WHERE balance > 100000;",
    "watsonx": "SELECT customer_id, name, balance FROM accounts WHERE balance > 100000;"
}

# Validate SQL outputs
sql_results = {
    "ollama": 125000.0,
    "watsonx": 125000.0
}
result_sql = validator.validate(
    sql_outputs, task_type="sql", sql_results=sql_results
)

print("\nSQL Generation Task")
print("=" * 60)
print(f"Similarity: {result_sql['similarity_scores']}")
print(f"Numeric result checks: {result_sql['task_validation']['result_match']}")

# Validate RAG outputs
rag_outputs = {
    "ollama": "Citigroup reported net credit losses of $1.2B in 2023. [citi_2024_10k]",
    "watsonx": "Citigroup reported net credit losses of $1.2 billion in 2023. [citi_2024_10k]"
}
rag_citations = {
    "ollama": ["citi_2024_10k"],
    "watsonx": ["citi_2024_10k"]
}

result_rag = validator.validate(
    rag_outputs, task_type="rag", citations=rag_citations
)

print("\nRAG Task")
print("=" * 60)
print(f"Similarity: {result_rag['similarity_scores']}")
print(
    "Citation sets matched:",
    result_rag["task_validation"]["citation_consistent"],
)

# Audit trail
print("\nCross-Provider Audit Report")
print("=" * 60)
for record in result_sql['audit_trail']:
    print(f"Providers: {record['providers']}")
    print(f"Output hashes: {record['output_hashes']}")
```

Run it:

```bash
python run_cross_provider_validation.py
```

## Step 4: Illustrative Numeric Tolerance (±5%)

This exercise uses a configurable **±5% tolerance** to demonstrate numeric
comparison. It is not a universal GAAP materiality threshold. Materiality is
context-dependent and must be set by the responsible accounting, risk, and
control functions.

**Example: Numeric comparison**

```python
def validate_numeric_tolerance(value1: float, value2: float, tolerance_pct: float = 5.0) -> bool:
    """Check whether two values meet an exercise-specific tolerance."""
    if value1 == 0 and value2 == 0:
        return True
    if value1 == 0 or value2 == 0:
        return False

    diff_pct = abs(value1 - value2) / max(value1, value2) * 100
    return diff_pct <= tolerance_pct

# Test cases
print(validate_numeric_tolerance(2.4, 2.5, tolerance_pct=5.0))  # True (4.0% diff)
print(validate_numeric_tolerance(100, 110, tolerance_pct=5.0))  # False (9.1% diff)
print(validate_numeric_tolerance(1000, 1040, tolerance_pct=5.0))  # True (3.8% diff)
```

**Why 5%?**
- It makes the comparison easy to inspect in this exercise
- It is configurable rather than a compliance rule
- Production tolerances should follow the task's approved control standard

The standalone helper and `CrossProviderValidator` both use a symmetric
max-denominator percentage, so reversing provider order does not change the
numeric comparison.

## Step 5: Multi-Run Cross-Provider Test

Compare five pairs of outputs collected separately from the two provider
configurations:

```python
#!/usr/bin/env python3
"""
Multi-run cross-provider consistency test.
"""
from harness.cross_provider_validation import CrossProviderValidator

validator = CrossProviderValidator(providers=["ollama", "watsonx"], tolerance_pct=5.0)

# Collect these with the provider clients before comparison. They are expanded
# here so the example remains runnable without unsupported validator kwargs.
ollama_outputs = [
    "SELECT customer_id, name, balance FROM accounts WHERE balance > 100000;"
] * 5
watsonx_outputs = [
    "SELECT customer_id, name, balance FROM accounts WHERE balance > 100000;"
] * 5
ollama_results = [125000.0] * 5
watsonx_results = [125000.0] * 5

results = []
for i, (ollama_output, watsonx_output) in enumerate(
    zip(ollama_outputs, watsonx_outputs), start=1
):
    result = validator.validate(
        outputs={"ollama": ollama_output, "watsonx": watsonx_output},
        task_type="sql",
        sql_results={
            "ollama": ollama_results[i - 1],
            "watsonx": watsonx_results[i - 1],
        },
    )
    query_match = all(
        score >= 0.95 for score in result["similarity_scores"].values()
    )
    numeric_match = all(
        result["task_validation"]["result_match"].values()
    )
    pair_match = query_match and numeric_match
    results.append(pair_match)
    print(f"Pair {i}: {'✅ Match' if pair_match else '⚠️ Review'}")

consistency_rate = sum(results) / len(results) * 100
print(f"\nObserved pair agreement: {consistency_rate:.0f}%")
```

**Expected output:**

```
Pair 1: ✅ Match
Pair 2: ✅ Match
Pair 3: ✅ Match
Pair 4: ✅ Match
Pair 5: ✅ Match

Observed pair agreement: 100%
```

## Step 6: Migration Decision Matrix

Use cross-provider replay to decide whether a migration needs more investigation.
Agreement in this exercise is evidence about repeatability, not a safety or
compliance determination:

| Scenario | Ollama → watsonx | Observed validation | Follow-up |
|----------|------------------|---------------------|-----------|
| **SQL (Tier 1 → Tier 1)** | Qwen2.5-7B → Granite-3-8B | 100% match | Path-aware replay |
| **RAG (Tier 1 → Tier 1)** | Qwen2.5-7B → Granite-3-8B | ≥95% match | Inspect retrieval evidence |
| **SQL (Tier 1 → Tier 2)** | Qwen2.5-7B → Llama-3.3-70B | 100% match | Path-aware replay |
| **RAG (Tier 1 → Tier 2)** | Qwen2.5-7B → Llama-3.3-70B | <95% match | Investigate variation |
| **Any (Tier 1 → Tier 3)** | Qwen2.5-7B → GPT-OSS-120B | <50% match | Full requalification |

**Repeatability triage:**

```python
def migration_replay_status(
    source_tier: int, target_tier: int, task_type: str
) -> str:
    """Prioritize replay work; do not treat this as a deployment gate."""
    if source_tier == 1 and target_tier == 1:
        return "run path-aware replay"

    if target_tier == 3:
        return "full requalification"

    if target_tier == 2 and task_type in ["sql", "summarize"]:
        return "run path-aware replay"

    return "investigate before relying on equivalent replay"

# Examples
print(migration_replay_status(1, 1, "rag"))
print(migration_replay_status(1, 2, "sql"))
print(migration_replay_status(1, 2, "rag"))
print(migration_replay_status(1, 3, "sql"))
```

## Understanding Provider Differences

Even with identical model versions, providers may differ in:

1. **Infrastructure**: GPU hardware, CUDA versions
2. **Quantization**: Different precision (FP16, FP32, INT8)
3. **Batching**: Request handling and parallelization
4. **Load balancing**: Multiple model replicas

!!! info "Replay the Exact Stack"
    Provider differences can reflect hardware, quantization, batching, routing,
    or model revisions. The workshop tiers do not isolate those causes; compare
    the exact source and target configurations.

## Troubleshooting

### watsonx.ai Connection Issues

```python
# Test watsonx.ai connectivity
from ibm_watsonx_ai.foundation_models import ModelInference
import os

try:
    model = ModelInference(
        model_id="ibm/granite-3-8b-instruct",
        api_key=os.getenv("WATSONX_API_KEY"),
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        url=os.getenv("WATSONX_URL")
    )
    print("✅ watsonx.ai connection successful")
except Exception as e:
    print(f"❌ watsonx.ai connection failed: {e}")
```

### Similarity Below 95%

If cross-provider similarity is unexpectedly low:

1. **Check model versions**: Ensure same base model
2. **Verify temperature**: Must be exactly 0.0
3. **Use explicit seeds**: Set `seed=42` for both
4. **Inspect raw outputs**: Look for formatting differences

```python
# Debug output differences
print("Ollama output:", repr(ollama_output))
print("watsonx output:", repr(watsonx_output))
```

## Key Takeaways

1. **Cross-provider validation** measures whether replay evidence changes across serving stacks
2. **Tier 1 configurations** showed high agreement in the bounded workshop runs
3. **The ±5% example** illustrates a configurable numeric tolerance
4. **Framework's `CrossProviderValidator`** automates testing
5. **Replay records** document the compared outputs and observed agreement

## Quiz: Test Your Understanding

??? question "What does the ±5% value represent in this lab?"
    **Answer**: An illustrative, configurable comparison tolerance. It is not a
    universal GAAP materiality threshold.

??? question "Why do Tier 1 models show better cross-provider consistency?"
    **Answer**: This exercise does not isolate the cause. Model family, serving
    stack, task, decoding, and hardware all differ and should be tested rather
    than inferred from parameter count.

??? question "What should happen before treating two provider configurations as equivalent?"
    **Answer**: Run the intended task under a frozen replay configuration,
    inspect decision and path evidence, and apply the workflow's approved
    correctness and control checks.

## Next Steps

Now that you understand cross-provider validation:

1. **Proceed to [Lab 6: Extending the Framework](../lab-6/README.md)** to add custom tasks
2. Test your own provider combinations
3. Review `harness/cross_provider_validation.py` for implementation details

---

!!! success "Lab 5 Complete!"
    You can now compare captured outputs across providers and identify where a
    migration needs deeper review. Ready to customize the framework? Move on to
    [Lab 6: Extending the Framework](../lab-6/README.md)!
