# API Reference

Complete reference documentation for the Output Drift framework components.

## Core Classes

### DeterministicRetriever

**Location**: `harness/deterministic_retriever.py`

Ensures reproducible SEC 10-K retrieval with multi-key ordering.

```python
from harness.deterministic_retriever import create_retriever_from_files

retriever = create_retriever_from_files(
    corpus_path="data/sec",
    chunk_size=200,
    overlap=50
)
```

**Methods**:

#### `retrieve(query, k=5)`

Retrieve top-k chunks with deterministic ordering.

**Parameters**:
- `query` (str): Search query
- `k` (int, default=5): Number of chunks to return

**Returns**:
- `List[Tuple[str, str, Dict]]`: List of `(snippet_id, text, metadata)` tuples

**Example**:
```python
results = retriever.retrieve("What were net credit losses?", k=5)
for snippet_id, text, metadata in results:
    print(f"{snippet_id}: {text[:100]}...")
```

---

### CrossProviderValidator

**Location**: `harness/cross_provider_validation.py`

Validates consistency across local (Ollama) and cloud (watsonx.ai) providers.

```python
from harness.cross_provider_validation import CrossProviderValidator

validator = CrossProviderValidator(
    providers=["ollama", "watsonx"],
    tolerance_pct=5.0  # illustrative; configure for the task
)
```

**Methods**:

#### `validate(outputs, task_type, citations=None, sql_results=None)`

Validate output consistency across providers using pre-collected outputs.

**Parameters**:
- `outputs` (Dict[str, str]): Provider name to output text mapping
- `task_type` (str): One of `"rag"`, `"sql"`, `"summary"`
- `citations` (Dict[str, List[str]], optional): Provider to citation list (for RAG)
- `sql_results` (Dict[str, Any], optional): Provider to numeric result (for SQL)

**Returns**:
- Dict with keys:
    - `consistent` (bool): Whether outputs match within tolerance
    - `similarity_scores` (Dict[str, float]): Pairwise similarity scores
    - `task_validation` (Dict): Task-specific validation details
    - `audit_trail` (List[Dict]): Validation audit records

**Example**:
```python
# Collect outputs from each provider first, then validate
outputs = {
    "ollama": "SELECT customer_id FROM accounts WHERE balance > 100000",
    "watsonx": "SELECT customer_id FROM accounts WHERE balance > 100000"
}
result = validator.validate(outputs, task_type="sql")
print(f"Consistent: {result['consistent']}")
print(f"Similarity: {result['similarity_scores']}")
```

---

## Task Definitions

**Location**: `harness/task_definitions.py` (formatting functions) and `run_evaluation.py` (prompt templates)

### RAG Task

```json
{
  "rag": {
    "description": "RAG Q&A over SEC 10-K filings with citation validation",
    "prompts": [...],
    "system_prompt": "You are a precise financial analyst...",
    "temperature": 0.0,
    "seed": 42
  }
}
```

### SQL Task

```json
{
  "sql": {
    "description": "Text-to-SQL with a configurable numeric tolerance",
    "prompts": [...],
    "system_prompt": "You write SQLite SQL ONLY...",
    "schema_description": "transactions(id INT, date TEXT, region TEXT, amount REAL, category TEXT)",
    "temperature": 0.0,
    "seed": 42
  }
}
```

### Summarization Task

```json
{
  "summary": {
    "description": "Policy-bounded JSON summarization with schema constraints",
    "prompts": [...],
    "system_prompt": "You produce STRICT JSON...",
    "schema": {...},
    "temperature": 0.0,
    "seed": 42
  }
}
```

---

## Configuration

### Environment Variables

Create `.env` file in repository root:

```bash
# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434

# IBM watsonx.ai
WATSONX_API_KEY=your_api_key
WATSONX_PROJECT_ID=your_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# Anthropic (optional)
ANTHROPIC_API_KEY=your_key

# Google Gemini (optional)
GEMINI_API_KEY=your_key
```

---

## Audit Trail Format

**Location**: `traces/*.jsonl`

Each line is a JSON object:

```json
{
  "timestamp": "2025-11-07T14:23:45.123Z",
  "run_id": "experiment_001",
  "model": "qwen2.5:7b-instruct",
  "provider": "ollama",
  "temperature": 0.0,
  "seed": 42,
  "task_type": "sql",
  "prompt": "Generate SQL...",
  "response": "SELECT ...",
  "prompt_hash": "sha256:...",
  "response_hash": "sha256:...",
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

---

## Common Patterns

### Running Experiments

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Run with deterministic settings
response = client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
    seed=42
)
```

### Calculating Consistency

```python
import json
from collections import Counter

with open("traces/experiment.jsonl") as f:
    traces = [json.loads(line) for line in f]

response_hashes = [t["response_hash"] for t in traces]
counts = Counter(response_hashes)
most_common_count = counts.most_common(1)[0][1]
consistency_pct = (most_common_count / len(response_hashes)) * 100

print(f"Consistency: {consistency_pct:.1f}%")
```

### Cross-Provider Comparison

```python
from rapidfuzz.distance import Levenshtein

distance = Levenshtein.normalized_distance(output1, output2)
similarity = 1.0 - distance

print(f"Similarity: {similarity:.1%}")
print(f"Match: {similarity >= 0.95}")
```

---

## Metrics Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Consistency** | `(identical_runs / total_runs) * 100` | % of runs producing same output |
| **Mean Drift** | `avg(Jaccard_distance(response_i, response_j))` | Average token-level difference |
| **Similarity** | `1.0 - Levenshtein.normalized_distance(s1, s2)` | Edit distance similarity |
| **Schema Validity** | JSON schema validation pass/fail | Structured output compliance |

---

## Further Reading

- [Lab 2: Framework Components](../lab-2/README.md)
- [Lab 5: Cross-Provider Testing](../lab-5/README.md)
- [GitHub Repository](https://github.com/ibm-client-engineering/output-drift-financial-llms)
