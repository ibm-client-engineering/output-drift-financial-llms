# Troubleshooting Guide

Common issues and solutions for the Output Drift framework.

## Installation Issues

### Python Version Mismatch

**Problem**: ImportError or syntax errors

**Solution**:
```bash
# Check Python version (must be 3.11+)
python --version

# Use specific version if multiple installed
python3.11 -m venv venv
source venv/bin/activate
```

### Dependency Conflicts

**Problem**: Package version conflicts

**Solution**:
```bash
# Fresh virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Ollama Issues

### Connection Refused

**Problem**: `ConnectionRefusedError: [Errno 61] Connection refused`

**Solution**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# Pull model if missing
ollama pull qwen2.5:7b-instruct
```

### Model Not Found

**Problem**: `Model 'qwen2.5:7b-instruct' not found`

**Solution**:
```bash
# List available models
ollama list

# Pull the model
ollama pull qwen2.5:7b-instruct

# Verify
ollama list | grep qwen
```

### Slow Performance

**Problem**: Ollama responses taking >10 seconds

**Solution**:
```bash
# Check system resources
top

# Ensure GPU is being used (if available)
ollama show qwen2.5:7b-instruct | grep parameters

# Reduce concurrency if CPU-only
--concurrency 4  # instead of 16
```

---

## watsonx.ai Issues

### Authentication Failed

**Problem**: `401 Unauthorized` or `Invalid API key`

**Solution**:
```python
# Test credentials
import os
from dotenv import load_dotenv
load_dotenv()

print(f"API Key: {os.getenv('WATSONX_API_KEY')[:10]}...")
print(f"Project ID: {os.getenv('WATSONX_PROJECT_ID')}")
print(f"URL: {os.getenv('WATSONX_URL')}")

# Verify all are set
assert all([os.getenv('WATSONX_API_KEY'),
            os.getenv('WATSONX_PROJECT_ID'),
            os.getenv('WATSONX_URL')]), "Missing credentials"
```

**Check**:
1. `.env` file exists in repository root
2. No extra spaces or quotes in `.env`
3. API key has not expired
4. Project ID is correct

### Rate Limiting

**Problem**: `429 Too Many Requests`

**Solution**:
```python
# Add delay between requests
import time

for i in range(16):
    response = model.generate_text(prompt, params)
    time.sleep(1)  # 1 second delay
```

Or use built-in rate limiting:
```bash
python run_evaluation.py \
  --rate-limit 10 \  # Max 10 requests/minute
  --retry-delay 5    # 5 seconds between retries
```

---

## Database Issues

### SQLite Not Found

**Problem**: `sqlite3.OperationalError: no such table: transactions`

**Solution**:
```bash
# Regenerate database
python data/generate_toy_finance.py

# Verify tables
sqlite3 data/toy_finance.sqlite "SELECT name FROM sqlite_master WHERE type='table';"
```

### Schema Mismatch

**Problem**: SQL queries fail with schema errors

**Solution**:
```bash
# Check schema
sqlite3 data/toy_finance.sqlite ".schema transactions"

# Expected output:
# CREATE TABLE transactions(
#   id INTEGER PRIMARY KEY,
#   date TEXT,
#   region TEXT,
#   amount REAL,
#   category TEXT
# );
```

---

## Drift Detection Issues

### False Positives (Detecting Drift When There Isn't Any)

**Problem**: Tier 1 models showing <100% consistency

**Causes**:
1. **Non-deterministic seed**: Ensure `seed=42` is set
2. **Different model versions**: Check `ollama show model`
3. **System prompt variations**: Use exact prompts from templates
4. **Whitespace differences**: Normalize before comparison

**Solution**:
```python
# Normalize outputs before comparison
import re

def normalize(text: str) -> str:
    """Remove extra whitespace and normalize case."""
    return re.sub(r'\s+', ' ', text.strip().lower())

output1_norm = normalize(output1)
output2_norm = normalize(output2)
match = (output1_norm == output2_norm)
```

### False Negatives (Not Detecting Real Drift)

**Problem**: Drift exists but isn't detected

**Causes**:
1. **n too small**: Must use n≥16 for reliable detection
2. **Hash collisions**: Unlikely but possible with SHA-256
3. **Semantic drift not captured**: Same tokens, different meaning

**Solution**:
```bash
# Always use n=16 (as in paper)
--concurrency 16

# Additionally check factual consistency (for RAG)
--validate-facts true
```

---

## API Errors

### OpenAI Client Issues

**Problem**: `openai.APIError` or similar

**Solution**:
```python
# For Ollama, use OpenAI client with custom base_url
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",  # Note: /v1 suffix
    api_key="ollama"  # Ollama doesn't check API key
)
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'harness'`

**Solution**:
```bash
# Ensure you're in the repository root
cd /path/to/output-drift-financial-llms

# Activate virtual environment
source venv/bin/activate

# Verify installation
python -c "import harness; print('✅ Import successful')"
```

---

## Performance Issues

### Memory Errors

**Problem**: `MemoryError` or system freezing

**Solution**:
```bash
# Reduce concurrency
--concurrency 4  # Instead of 16

# Process in batches
--batch-size 4

# Use smaller model
qwen2.5:7b-instruct  # Instead of larger models
```

### Slow Experiments

**Problem**: Experiments taking >1 hour

**Solution**:
```bash
# Profile time per request
time curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "qwen2.5:7b-instruct", "prompt": "test"}'

# If >5 seconds per request:
# 1. Check CPU/GPU utilization
# 2. Reduce model size
# 3. Increase system resources
```

---

## Cross-Provider Issues

### Outputs Don't Match

**Problem**: Ollama and watsonx outputs differ significantly

**Expected**: Tier 1 → Tier 1 should match ≥95%

**Debugging**:
```python
# Compare raw outputs
print("Ollama:", repr(ollama_output))
print("watsonx:", repr(watsonx_output))

# Check lengths
print(f"Ollama length: {len(ollama_output)}")
print(f"watsonx length: {len(watsonx_output)}")

# Calculate similarity
from rapidfuzz.distance import Levenshtein
sim = 1.0 - Levenshtein.normalized_distance(ollama_output, watsonx_output)
print(f"Similarity: {sim:.1%}")
```

**Common causes**:
1. Different model versions (Qwen vs Granite)
2. Temperature not exactly 0.0
3. System prompts differ
4. Different tokenization

---

## Compliance Validation Errors

### Schema Validation Fails

**Problem**: `jsonschema.ValidationError`

**Solution**:
```python
import json
from jsonschema import validate, ValidationError

# Test JSON parsing
try:
    data = json.loads(response)
    print("✅ Valid JSON")
except json.JSONDecodeError as e:
    print(f"❌ Invalid JSON: {e}")

# Test schema validation
try:
    validate(data, schema)
    print("✅ Schema valid")
except ValidationError as e:
    print(f"❌ Schema invalid: {e.message}")
```

---

## Common Error Messages

### `FileNotFoundError: [Errno 2] No such file or directory: 'traces/'`

**Solution**:
```bash
mkdir -p traces
```

### `RuntimeError: Found no NVIDIA driver on your system`

**Not an error** - Ollama will use CPU, which is fine for 7-8B models.

### `ImportError: cannot import name 'DeterministicRetriever'`

**Solution**:
```bash
# Ensure harness/__init__.py exists
ls harness/__init__.py

# Reinstall if missing
pip install -e .
```

---

## Getting Help

If you're still stuck:

1. **Check Logs**:
   ```bash
   # Ollama logs
   ollama logs

   # Python logs
   python run_evaluation.py --verbose
   ```

2. **GitHub Issues**: [Open an issue](https://github.com/ibm-client-engineering/output-drift-financial-llms/issues/new)

3. **Email Support**: Contact maintainers (see README)

4. **Review Documentation**:
   - [API Reference](api.md)
   - [Lab Guides](../pre-work/README.md)
   - [Research Paper](paper.md)

---

## Quick Diagnostics Script

Run this to check your environment:

```python
#!/usr/bin/env python3
"""Quick diagnostics for Output Drift framework."""
import sys
import os
import subprocess

print("🔍 Diagnostics\n" + "=" * 60)

# Python version
print(f"Python: {sys.version}")
assert sys.version_info >= (3, 11), "❌ Python 3.11+ required"
print("✅ Python version OK\n")

# Dependencies
try:
    import openai, pandas, matplotlib
    print("✅ Dependencies installed\n")
except ImportError as e:
    print(f"❌ Missing dependency: {e}\n")

# Ollama
try:
    result = subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"],
                          capture_output=True, timeout=5)
    if result.returncode == 0:
        print("✅ Ollama running\n")
    else:
        print("⚠️  Ollama not responding\n")
except Exception as e:
    print(f"❌ Ollama check failed: {e}\n")

# Environment variables
env_vars = ["WATSONX_API_KEY", "WATSONX_PROJECT_ID"]
for var in env_vars:
    if os.getenv(var):
        print(f"✅ {var} set")
    else:
        print(f"⚠️  {var} not set")

print("\n" + "=" * 60)
print("Diagnostics complete!")
```

Save as `diagnostics.py` and run:
```bash
python diagnostics.py
```
