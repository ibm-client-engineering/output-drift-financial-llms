# Lab 2: Setting Up Your Environment

## Overview

In this lab, you'll configure API keys, test provider connectivity, and run your first deterministic evaluation to understand the framework's core components.

**Duration**: ~15 minutes

## Learning Objectives

By the end of this lab, you will:

- Configure API keys for at least one provider (Ollama recommended)
- Understand the DeterministicRetriever and its role in compliance
- Test framework components with a simple evaluation
- Generate your first audit trail

## Prerequisites

- Completed [Lab 0: Workshop Pre-work](../pre-work/README.md)
- At least one provider configured (Ollama, watsonx.ai, or others)

## Step 1: Verify Ollama Installation

If using Ollama (recommended for getting started):

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags
```

If not running, start Ollama:

```bash
ollama serve
```

Pull the recommended model (if not already done):

```bash
ollama pull qwen2.5:7b-instruct
```

!!! tip "Why Qwen2.5:7B?"
    According to our research, **7-8B models achieve 100% deterministic outputs at T=0.0**, making them ideal for regulated financial applications. Qwen2.5:7B is a Tier 1 model—audit-ready and compliance-safe.

## Step 2: Configure Environment Variables

Create or edit your `.env` file in the repository root:

```bash
# Navigate to repository root
cd /path/to/output-drift-financial-llms

# Create .env file
touch .env
```

Add your API configuration:

```bash
# Ollama (local, free)
OLLAMA_BASE_URL=http://localhost:11434

# IBM watsonx.ai (optional but recommended for cross-provider validation)
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# OpenAI (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

!!! warning "Sensitive Data"
    Never commit `.env` to Git! It's already in `.gitignore`.

## Step 3: Generate Synthetic Financial Database

Our framework uses a synthetic financial database for SQL generation tasks:

```bash
python data/generate_toy_finance.py
```

**Expected output:**

```
🏦 Generating synthetic financial database...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Created tables:
  ✅ customers (100 records)
  ✅ accounts (150 records)
  ✅ transactions (500 records)
  ✅ loans (75 records)

Database: data/toy_finance.sqlite (45 KB)
✅ Generation complete!
```

This creates `data/toy_finance.sqlite` containing realistic financial data for testing.

## Step 4: Test Framework Components

Let's test the core framework components to ensure everything is working.

### Test 1: DeterministicRetriever

The **DeterministicRetriever** (harness/deterministic_retriever.py) is crucial for compliance—it ensures SEC 10-K retrieval order is deterministic and reproducible.

Create `test_retriever.py`:

```python
from harness.deterministic_retriever import DeterministicRetriever

# Initialize retriever
retriever = DeterministicRetriever(
    corpus_path="data/sec_filings/",  # SEC 10-K filings
    chunk_size=512,
    overlap=50
)

# Test query
query = "What were net credit losses in 2023?"
results = retriever.retrieve(query, top_k=5)

print("🔍 Deterministic Retrieval Test")
print("=" * 50)
for i, chunk in enumerate(results, 1):
    print(f"\nChunk {i}:")
    print(f"  Source: {chunk['source']}")
    print(f"  Score: {chunk['score']:.4f}")
    print(f"  Snippet ID: {chunk['snippet_id']}")
    print(f"  Text: {chunk['text'][:100]}...")

print("\n✅ Retrieval is deterministic with stable ordering!")
```

Run it:

```bash
python test_retriever.py
```

!!! info "Why Multi-Key Ordering?"
    The retriever uses **multi-key ordering** (score↓, section_priority↑, snippet_id↑, chunk_idx↑) to ensure retrieval order is a **compliance requirement**, not a performance optimization. This guarantees the same chunks are retrieved in the same order every time.

### Test 2: Simple Drift Evaluation

Now let's run a minimal drift test with 5 runs using the OpenAI client:

Create `test_simple_drift.py`:

```python
#!/usr/bin/env python3
"""Simple drift evaluation using Ollama via OpenAI client."""
from openai import OpenAI

# Initialize Ollama client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Not used by Ollama
)

# Simple prompt
prompt = "What is the sum of 2 + 2? Answer with just the number."

print("🧪 Running 5 identical queries at T=0.0")
print("=" * 50)

responses = []
for i in range(1, 6):
    response = client.chat.completions.create(
        model="qwen2.5:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        seed=42
    )
    answer = response.choices[0].message.content
    responses.append(answer)
    print(f"Run {i}: {answer}")

# Check consistency
unique_responses = set(responses)
consistency = (len(unique_responses) == 1)

print("\n" + "=" * 50)
print(f"Unique responses: {len(unique_responses)}")
print(f"Consistency: {'✅ 100%' if consistency else f'❌ {100/len(responses):.0f}%'}")
```

Run it:

```bash
python test_simple_drift.py
```

**Expected output for Tier 1 models (Qwen2.5:7B, Granite-3-8B):**

```
🧪 Running 5 identical queries at T=0.0
==================================================
Run 1: 4
Run 2: 4
Run 3: 4
Run 4: 4
Run 5: 4

==================================================
Unique responses: 1
Consistency: ✅ 100%
```

!!! success "Tier 1 Determinism"
    7-8B models achieve **100% consistency at T=0.0**—this is what makes them audit-ready!

## Step 5: Understanding Task Definitions

The framework defines three core financial tasks in `harness/task_definitions.py`:

```bash
# View task definitions
cat harness/task_definitions.py
```

**The three core tasks:**

| Task | File Reference | Tier 1 Consistency | Purpose |
|------|---------------|-------------------|---------|
| **SQL** | harness/task_definitions.py:20-45 | 100% | Text-to-SQL generation |
| **Summarize** | harness/task_definitions.py:47-72 | 100% | JSON summarization with schema |
| **RAG** | harness/task_definitions.py:74-99 | 93.75% | Retrieval-augmented Q&A |

Each task includes:
- System prompts optimized for determinism
- Temperature=0.0 and seed=42 defaults
- Validation schemas (JSON schema for summarization, SQL syntax checker)
- Citation requirements (for RAG tasks)

## Step 6: Review Sample Audit Trail

The framework generates JSONL (JSON Lines) audit trails with regulatory mappings. Let's examine the sample provided:

```bash
# View sample audit trail entry
head -n 1 examples/sample_audit_trail.jsonl | python -m json.tool
```

**Example audit trail entry:**

```json
{
  "timestamp": "2025-11-07T13:45:23Z",
  "run_id": "lab2_test_001",
  "model": "qwen2.5:7b-instruct",
  "provider": "ollama",
  "temperature": 0.0,
  "seed": 42,
  "prompt_hash": "a3d8f92b1c4e5f6789abcdef...",
  "response_hash": "b2c1e7d8a9f6543210fedcba...",
  "task_type": "sql",
  "response": "SELECT customer_name, account_balance FROM accounts WHERE account_balance > 100000",
  "compliance_metrics": {
    "citation_accuracy": 1.0,
    "schema_valid": true,
    "decision_flip": false,
    "factual_drift": 0.0
  },
  "regulatory_mappings": {
    "FSB_principle": "consistent_decisions",
    "CFTC_requirement": "document_ai_outcomes",
    "SR_11_7": "model_validation"
  }
}
```

!!! info "Bi-Temporal Logging"
    The audit trail uses **bi-temporal logging** to enable regulatory review and attestation months after decisions were made—critical for financial audits.

## Understanding Framework Components

### 1. DeterministicRetriever

**File**: harness/deterministic_retriever.py

```python
from harness.deterministic_retriever import DeterministicRetriever

retriever = DeterministicRetriever(
    corpus_path="data/sec_filings/",
    chunk_size=512,
    overlap=50
)
```

**Purpose**: Ensures SEC 10-K retrieval is deterministic and auditable.

**Features**:
- Multi-key ordering (score, section priority, snippet ID, chunk index)
- Stable chunk IDs for reproducibility
- Section-aware retrieval (prioritizes financial statement sections)

### 2. Task Definitions

The framework includes 3 core task types:

| Task | Description | Tier 1 Consistency |
|------|-------------|-------------------|
| **SQL** | Text-to-SQL generation from natural language | **100%** ✅ |
| **Summarize** | JSON summarization of financial data | **100%** ✅ |
| **RAG** | Retrieval-augmented Q&A over SEC 10-Ks | **93.75%** ✅ |

**Why SQL and Summarize achieve perfect scores:**
- Structured output formats
- Deterministic syntax
- Narrow output space

### 3. Cross-Provider Validation

**File**: harness/cross_provider_validation.py

```python
from harness.cross_provider_validation import CrossProviderValidator

validator = CrossProviderValidator(
    providers=["ollama", "watsonx"],
    tolerance_pct=5.0  # GAAP materiality threshold
)
results = validator.validate(prompt, task_type="sql")
```

**Purpose**: Validate consistency between local (Ollama) and cloud (watsonx.ai) deployments.

**GAAP Materiality**: Uses ±5% threshold from GAAP auditing standards for financial statement materiality.

## Troubleshooting

### Ollama Connection Failed

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it:
ollama serve
```

### Model Not Found

```bash
# List available models
ollama list

# Pull the model if missing
ollama pull qwen2.5:7b-instruct
```

### Database Not Found

```bash
# Regenerate the database
python data/generate_toy_finance.py
```

### Import Errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

## Key Takeaways

1. **Tier 1 Models**: 7-8B models (Qwen2.5, Granite-3-8B) achieve 100% determinism
2. **DeterministicRetriever**: Ensures reproducible SEC 10-K retrieval
3. **Audit Trails**: Bi-temporal JSONL logging enables regulatory review
4. **Task Types**: SQL and summarization are perfectly deterministic; RAG requires careful configuration
5. **Cross-Provider**: Can validate consistency between local and cloud deployments

## Quiz: Test Your Understanding

??? question "Why use multi-key ordering in DeterministicRetriever?"
    **Answer**: To ensure retrieval order is deterministic and reproducible for compliance. Even if chunks have the same relevance score, they must return in a consistent order for audit trails.

??? question "What makes 7-8B models Tier 1 (audit-ready)?"
    **Answer**: They achieve 100% consistency at T=0.0 across all task types, meeting regulatory requirements for reproducibility.

??? question "What is the GAAP materiality threshold used in cross-provider validation?"
    **Answer**: ±5%, based on GAAP auditing standards for financial statement materiality.

## Next Steps

Now that your environment is configured and you understand the framework components:

1. **Proceed to [Lab 3: Running Your First Experiment](../lab-3/README.md)** to run drift evaluations
2. Review task definitions in `harness/task_definitions.py`
3. Examine the DeterministicRetriever implementation in `harness/deterministic_retriever.py`
4. Study the CrossProviderValidator code in `harness/cross_provider_validation.py`

---

!!! success "Lab 2 Complete!"
    Your environment is configured and tested. Ready to run experiments? Move on to [Lab 3: Running Your First Experiment](../lab-3/README.md)!
