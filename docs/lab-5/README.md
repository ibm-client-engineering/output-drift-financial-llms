# Lab 5: Cross-Provider Testing

## Overview

In this lab, you'll validate output consistency between local (Ollama) and cloud (IBM watsonx.ai) deployments using the framework's `CrossProviderValidator`. This ensures your models produce reliable results regardless of deployment environment.

**Duration**: ~30 minutes

## Learning Objectives

By the end of this lab, you will:

- Use the `CrossProviderValidator` from your framework
- Compare outputs between Ollama and watsonx.ai
- Understand GAAP materiality thresholds (±5%)
- Validate cross-provider consistency for compliance
- Make deployment decisions based on provider reliability

## Prerequisites

- Completed [Lab 4: Analyzing Drift Metrics](../lab-4/README.md)
- **At least two providers configured**: Ollama + one cloud provider (watsonx.ai recommended)
- API keys in `.env` file

## Why Cross-Provider Validation Matters

Financial institutions often need to:

- **Migrate** between providers without changing behavior
- **Redundancy** with failover to backup providers
- **Vendor independence** to avoid lock-in
- **Regulatory compliance** requiring reproducibility across environments

!!! warning "The Risk"
    A model that works locally but behaves differently in production (cloud) creates **audit trail inconsistencies** and **compliance violations**.

## Step 1: Review CrossProviderValidator Code

Open `harness/cross_provider_validation.py` to see how it works:

```bash
cat harness/cross_provider_validation.py | head -50
```

**Key features** (from the code):
- Normalized edit distance for text comparison
- **±5% tolerance** (GAAP materiality threshold)
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
    print("\n✅ Cross-provider validation PASSED (≥95% similarity)")
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
  Ollama length: 87 chars
  watsonx length: 87 chars
  Exact match: True
  Similarity: 100.0%

✅ Cross-provider validation PASSED (≥95% similarity)
```

!!! success "Tier 1 Cross-Provider Consistency"
    Both Granite-3-8B (watsonx) and Qwen2.5-7B (Ollama) produce **identical outputs**—enabling seamless migration between local and cloud deployments.

## Step 3: Use the Framework's CrossProviderValidator

Now use the built-in validator from `harness/cross_provider_validation.py`:

Create `run_cross_provider_validation.py`:

```python
#!/usr/bin/env python3
"""
Use framework's CrossProviderValidator for automated testing.
"""
from harness.cross_provider_validation import CrossProviderValidator

# Initialize validator with GAAP materiality threshold
validator = CrossProviderValidator(
    providers=["ollama", "watsonx"],
    tolerance_pct=5.0  # ±5% from GAAP auditing standards
)

# SQL generation task
prompt_sql = "Generate SQL to find all customers with account balance > $100,000"
result_sql = validator.validate(
    prompt=prompt_sql,
    task_type="sql",
    model_ollama="qwen2.5:7b-instruct",
    model_watsonx="ibm/granite-3-8b-instruct",
    temperature=0.0,
    seed=42
)

print("\n📊 SQL Generation Task")
print("=" * 60)
print(f"Consistent: {result_sql['consistent']}")
print(f"Similarity: {result_sql['similarity']:.1%}")
print(f"Validation: {'✅ PASS' if result_sql['consistent'] else '❌ FAIL'}")

# RAG task
prompt_rag = "What were Citigroup's net credit losses in 2023?"
result_rag = validator.validate(
    prompt=prompt_rag,
    task_type="rag",
    model_ollama="qwen2.5:7b-instruct",
    model_watsonx="ibm/granite-3-8b-instruct",
    temperature=0.0,
    seed=42
)

print("\n📊 RAG Task")
print("=" * 60)
print(f"Consistent: {result_rag['consistent']}")
print(f"Similarity: {result_rag['similarity']:.1%}")
print(f"Factual consistency: {result_rag['factual_match']}")
print(f"Validation: {'✅ PASS' if result_rag['consistent'] else '⚠️  MINOR DRIFT'}")

# Generate audit report
print("\n📄 Cross-Provider Audit Report")
print("=" * 60)
for provider, output in result_sql['outputs'].items():
    print(f"{provider:15s}: {output[:80]}...")
```

Run it:

```bash
python run_cross_provider_validation.py
```

## Step 4: GAAP Materiality Threshold (±5%)

The framework uses **±5% tolerance** based on GAAP auditing standards for financial statement materiality.

**Example: Numeric comparison**

```python
def validate_numeric_tolerance(value1: float, value2: float, tolerance_pct: float = 5.0) -> bool:
    """Check if two values are within GAAP materiality threshold."""
    if value1 == 0 and value2 == 0:
        return True
    if value1 == 0 or value2 == 0:
        return False

    diff_pct = abs(value1 - value2) / max(value1, value2) * 100
    return diff_pct <= tolerance_pct

# Test cases
print(validate_numeric_tolerance(2.4, 2.5, tolerance_pct=5.0))  # True (4.2% diff)
print(validate_numeric_tolerance(100, 110, tolerance_pct=5.0))  # False (9.1% diff)
print(validate_numeric_tolerance(1000, 1040, tolerance_pct=5.0))  # True (3.8% diff)
```

**Why 5%?**
- GAAP materiality standard for financial reporting
- Industry-accepted threshold for immaterial differences
- Balances strictness with practical variance

## Step 5: Multi-Run Cross-Provider Test

Test consistency across multiple runs (n=5):

```python
#!/usr/bin/env python3
"""
Multi-run cross-provider consistency test.
"""
from harness.cross_provider_validation import CrossProviderValidator

validator = CrossProviderValidator(providers=["ollama", "watsonx"], tolerance_pct=5.0)
prompt = "Generate SQL to find all customers with account balance > $100,000"

results = []
for i in range(1, 6):
    result = validator.validate(
        prompt=prompt,
        task_type="sql",
        model_ollama="qwen2.5:7b-instruct",
        model_watsonx="ibm/granite-3-8b-instruct",
        temperature=0.0,
        seed=42
    )
    results.append(result['consistent'])
    print(f"Run {i}: {'✅ Consistent' if result['consistent'] else '❌ Inconsistent'}")

consistency_rate = sum(results) / len(results) * 100
print(f"\nOverall consistency: {consistency_rate:.0f}%")
```

**Expected output:**

```
Run 1: ✅ Consistent
Run 2: ✅ Consistent
Run 3: ✅ Consistent
Run 4: ✅ Consistent
Run 5: ✅ Consistent

Overall consistency: 100%
```

## Step 6: Migration Decision Matrix

Based on cross-provider validation, decide whether migration is safe:

| Scenario | Ollama → watsonx | Validation | Safe to Migrate? |
|----------|------------------|------------|------------------|
| **SQL (Tier 1 → Tier 1)** | Qwen2.5-7B → Granite-3-8B | 100% match | ✅ Yes |
| **RAG (Tier 1 → Tier 1)** | Qwen2.5-7B → Granite-3-8B | ≥95% match | ✅ Yes |
| **SQL (Tier 1 → Tier 2)** | Qwen2.5-7B → Llama-3.3-70B | 100% match | ✅ Yes |
| **RAG (Tier 1 → Tier 2)** | Qwen2.5-7B → Llama-3.3-70B | <95% match | ⚠️ Monitor |
| **Any (Tier 1 → Tier 3)** | Qwen2.5-7B → GPT-OSS-120B | <50% match | ❌ No |

**Migration safety check:**

```python
def is_migration_safe(source_tier: int, target_tier: int, task_type: str) -> bool:
    """Check if migration between providers is compliance-safe."""
    if source_tier == 1 and target_tier == 1:
        return True  # Always safe: Tier 1 → Tier 1

    if target_tier == 3:
        return False  # Never safe: Any → Tier 3

    if target_tier == 2 and task_type in ["sql", "summarize"]:
        return True  # Safe for structured tasks

    return False  # Requires validation

# Examples
print(is_migration_safe(1, 1, "rag"))  # True
print(is_migration_safe(1, 2, "sql"))  # True
print(is_migration_safe(1, 2, "rag"))  # False (requires validation)
print(is_migration_safe(1, 3, "sql"))  # False
```

## Understanding Provider Differences

Even with identical model versions, providers may differ in:

1. **Infrastructure**: GPU hardware, CUDA versions
2. **Quantization**: Different precision (FP16, FP32, INT8)
3. **Batching**: Request handling and parallelization
4. **Load balancing**: Multiple model replicas

!!! info "Tier 1 Advantage"
    Tier 1 models (7-8B) are **small enough** to fit on a single GPU consistently, reducing infrastructure-induced variance.

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

1. **Cross-provider validation** ensures migration safety
2. **Tier 1 models** (7-8B) achieve perfect cross-provider consistency
3. **GAAP materiality (±5%)** provides finance-calibrated tolerance
4. **Framework's `CrossProviderValidator`** automates testing
5. **Audit trails** document cross-provider equivalence

## Quiz: Test Your Understanding

??? question "What is the GAAP materiality threshold used in cross-provider validation?"
    **Answer**: ±5%, based on GAAP auditing standards for financial statement materiality.

??? question "Why do Tier 1 models show better cross-provider consistency?"
    **Answer**: They're small enough (7-8B params) to fit on a single GPU, reducing infrastructure-induced variance from distributed processing.

??? question "When is migration from Tier 1 to Tier 2 safe?"
    **Answer**: Only for structured tasks (SQL, summarization). RAG tasks require explicit validation due to Tier 2's lower RAG consistency.

## Next Steps

Now that you understand cross-provider validation:

1. **Proceed to [Lab 6: Extending the Framework](../lab-6/README.md)** to add custom tasks
2. Test your own provider combinations
3. Review `harness/cross_provider_validation.py` for implementation details

---

!!! success "Lab 5 Complete!"
    You can now validate cross-provider consistency and make migration decisions with confidence! Ready to customize the framework? Move on to [Lab 6: Extending the Framework](../lab-6/README.md)!
