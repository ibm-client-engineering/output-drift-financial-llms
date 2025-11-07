# Lab 6: Extending the Framework

## Overview

In this lab, you'll learn how to customize the framework for your own use cases by adding new tasks, modifying prompt templates, and integrating with your workflows.

**Duration**: ~30 minutes

## Learning Objectives

By the end of this lab, you will:

- Add custom tasks to `prompts/templates.json`
- Modify existing prompts for your domain
- Integrate the framework into CI/CD pipelines
- Create custom compliance validators
- Export results for regulatory reporting

## Prerequisites

- Completed [Lab 5: Cross-Provider Testing](../lab-5/README.md)
- Understanding of JSON structure
- Familiarity with your organization's compliance requirements

## Framework Architecture

The framework is designed for extensibility:

```
output-drift-financial-llms/
├── prompts/
│   └── templates.json          # ← Add your custom tasks here
├── harness/
│   ├── task_definitions.py     # Task execution logic
│   ├── deterministic_retriever.py
│   └── cross_provider_validation.py
├── data/
│   ├── sec_filings/            # ← Add your own documents
│   └── toy_finance.sqlite      # ← Use your database
└── examples/                   # ← Reference implementations
```

## Step 1: Understanding Template Structure

Open `prompts/templates.json` to see the existing tasks:

```bash
cat prompts/templates.json
```

**Current tasks:**
- `rag`: RAG Q&A over SEC 10-K filings
- `summary`: JSON summarization with schema validation
- `sql`: Text-to-SQL generation

Each task has:
- **description**: What the task does
- **prompts**: Array of test cases
- **system_prompt**: Instructions for the LLM
- **temperature**: 0.0 for determinism
- **seed**: 42 for reproducibility

## Step 2: Add a Custom Task - Credit Risk Analysis

Let's add a new task for credit risk assessment:

**Edit `prompts/templates.json`** and add after the `sql` section:

```json
{
  "credit_risk": {
    "description": "Credit risk classification with explainability requirements",
    "prompts": [
      {
        "id": "cr1",
        "profile": {
          "credit_score": 680,
          "income": 75000,
          "debt_to_income": 0.20,
          "employment_years": 5
        },
        "question": "Classify credit risk (LOW/MEDIUM/HIGH) and explain in one sentence.",
        "expected_risk": "LOW",
        "compliance_requirements": ["ECOA", "FCRA"]
      },
      {
        "id": "cr2",
        "profile": {
          "credit_score": 620,
          "income": 50000,
          "debt_to_income": 0.45,
          "employment_years": 1
        },
        "question": "Classify credit risk (LOW/MEDIUM/HIGH) and explain in one sentence.",
        "expected_risk": "MEDIUM",
        "compliance_requirements": ["ECOA", "FCRA"]
      }
    ],
    "system_prompt": "You are a fair and consistent credit risk analyst. Classify risk as LOW, MEDIUM, or HIGH. Provide a brief explanation in one sentence. Be consistent: identical inputs must always produce identical outputs for regulatory compliance.",
    "output_schema": {
      "type": "object",
      "properties": {
        "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
        "explanation": {"type": "string"}
      },
      "required": ["risk_level", "explanation"]
    },
    "temperature": 0.0,
    "seed": 42
  }
}
```

## Step 3: Create Task Executor for Custom Task

Create `custom_credit_risk.py`:

```python
#!/usr/bin/env python3
"""
Custom credit risk classification task with drift testing.
"""
import json
from openai import OpenAI

# Load custom task template
with open("prompts/templates.json") as f:
    templates = json.load(f)
    credit_risk_task = templates["credit_risk"]

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def run_credit_risk_assessment(profile: dict, model: str = "qwen2.5:7b-instruct", n_runs: int = 5):
    """Run credit risk assessment n times to test consistency."""
    prompt = f"""Profile:
- Credit Score: {profile['credit_score']}
- Annual Income: ${profile['income']:,}
- Debt-to-Income Ratio: {profile['debt_to_income']:.0%}
- Employment Years: {profile['employment_years']}

{credit_risk_task['prompts'][0]['question']}"""

    results = []
    for i in range(1, n_runs + 1):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": credit_risk_task['system_prompt']},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            seed=42
        )
        output = response.choices[0].message.content
        results.append(output)
        print(f"Run {i}: {output}")

    # Check consistency
    unique = len(set(results))
    consistency = (1 / unique) * 100 if unique > 0 else 100.0

    print(f"\n📊 Results:")
    print(f"  Total runs: {n_runs}")
    print(f"  Unique outputs: {unique}")
    print(f"  Consistency: {consistency:.0f}%")
    print(f"  Status: {'✅ Audit-ready' if consistency == 100 else '⚠️ Drift detected'}")

    return results

# Test with the first profile
profile1 = credit_risk_task['prompts'][0]['profile']
print("🧪 Testing Credit Risk Assessment\n")
results = run_credit_risk_assessment(profile1, n_runs=5)
```

Run it:

```bash
python custom_credit_risk.py
```

**Expected output (Tier 1 model):**

```
🧪 Testing Credit Risk Assessment

Run 1: {"risk_level": "LOW", "explanation": "Strong credit profile with good income-to-debt ratio and stable employment history."}
Run 2: {"risk_level": "LOW", "explanation": "Strong credit profile with good income-to-debt ratio and stable employment history."}
Run 3: {"risk_level": "LOW", "explanation": "Strong credit profile with good income-to-debt ratio and stable employment history."}
Run 4: {"risk_level": "LOW", "explanation": "Strong credit profile with good income-to-debt ratio and stable employment history."}
Run 5: {"risk_level": "LOW", "explanation": "Strong credit profile with good income-to-debt ratio and stable employment history."}

📊 Results:
  Total runs: 5
  Unique outputs: 1
  Consistency: 100%
  Status: ✅ Audit-ready
```

## Step 4: Add Domain-Specific Documents

To use RAG with your own documents:

1. **Add documents to `data/sec_filings/`** (or create a new folder):

```bash
mkdir -p data/custom_docs
```

2. **Update `deterministic_retriever.py`** to point to your folder:

```python
from harness.deterministic_retriever import DeterministicRetriever

retriever = DeterministicRetriever(
    corpus_path="data/custom_docs/",  # Your documents here
    chunk_size=512,
    overlap=50
)
```

3. **Test retrieval**:

```python
query = "What is our company's annual revenue?"
results = retriever.retrieve(query, top_k=5)
for i, chunk in enumerate(results, 1):
    print(f"Chunk {i}: {chunk['text'][:100]}...")
```

## Step 5: CI/CD Integration

Integrate drift testing into your CI/CD pipeline:

**Create `.github/workflows/drift-test.yml`**:

```yaml
name: LLM Output Drift Testing

on:
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight

jobs:
  drift-test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run drift evaluation
        env:
          WATSONX_API_KEY: ${{ secrets.WATSONX_API_KEY }}
          WATSONX_PROJECT_ID: ${{ secrets.WATSONX_PROJECT_ID }}
        run: |
          python run_evaluation.py \
            --model ibm/granite-3-8b-instruct \
            --temperature 0.0 \
            --concurrency 16 \
            --task sql \
            --output traces/ci_test.jsonl

      - name: Validate consistency
        run: |
          python -c "
          import json
          with open('traces/ci_test.jsonl') as f:
              data = [json.loads(line) for line in f]
          unique = len(set(d['response_hash'] for d in data))
          assert unique == 1, f'Drift detected: {unique} unique outputs'
          print('✅ Consistency check passed')
          "

      - name: Upload audit trail
        uses: actions/upload-artifact@v3
        with:
          name: drift-test-results
          path: traces/ci_test.jsonl
```

This pipeline:
- Runs on every PR and daily
- Tests for drift with n=16
- Fails CI if drift detected
- Uploads audit trails as artifacts

## Step 6: Custom Compliance Validator

Create a validator for your specific regulations:

**Create `custom_compliance_validator.py`**:

```python
#!/usr/bin/env python3
"""
Custom compliance validator for specific regulatory frameworks.
"""
import json
from typing import Dict, List

class CustomComplianceValidator:
    """
    Validate LLM outputs against custom compliance requirements.
    """

    def __init__(self, frameworks: List[str]):
        """
        Initialize validator.

        Args:
            frameworks: List of compliance frameworks (e.g., ["ECOA", "FCRA", "GDPR"])
        """
        self.frameworks = frameworks
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, callable]:
        """Load validation rules for each framework."""
        rules = {}

        # ECOA (Equal Credit Opportunity Act)
        if "ECOA" in self.frameworks:
            rules["ecoa_consistency"] = self._check_consistency
            rules["ecoa_no_discrimination"] = self._check_no_discrimination

        # FCRA (Fair Credit Reporting Act)
        if "FCRA" in self.frameworks:
            rules["fcra_explainability"] = self._check_explainability

        # GDPR
        if "GDPR" in self.frameworks:
            rules["gdpr_right_to_explanation"] = self._check_explainability
            rules["gdpr_data_minimization"] = self._check_data_minimization

        return rules

    def _check_consistency(self, outputs: List[str]) -> bool:
        """ECOA: Similar applicants must receive similar treatment."""
        unique_outputs = len(set(outputs))
        return unique_outputs == 1  # 100% consistency required

    def _check_no_discrimination(self, output: str) -> bool:
        """ECOA: No references to protected classes."""
        protected_terms = ["race", "gender", "age", "religion", "nationality"]
        return not any(term in output.lower() for term in protected_terms)

    def _check_explainability(self, output: str) -> bool:
        """FCRA/GDPR: Must include explanation."""
        return "explanation" in output.lower() or "because" in output.lower()

    def _check_data_minimization(self, output: str) -> bool:
        """GDPR: Don't expose unnecessary personal data."""
        pii_indicators = ["ssn", "social security", "passport", "driver license"]
        return not any(indicator in output.lower() for indicator in pii_indicators)

    def validate(self, outputs: List[str]) -> Dict[str, any]:
        """
        Run all validation rules.

        Args:
            outputs: List of LLM outputs to validate

        Returns:
            {
                "compliant": bool,
                "passed_rules": List[str],
                "failed_rules": List[str],
                "details": Dict[str, bool]
            }
        """
        results = {}
        for rule_name, rule_func in self.rules.items():
            if rule_name.endswith("_consistency"):
                results[rule_name] = rule_func(outputs)
            else:
                # Check all outputs
                results[rule_name] = all(rule_func(output) for output in outputs)

        passed = [k for k, v in results.items() if v]
        failed = [k for k, v in results.items() if not v]

        return {
            "compliant": len(failed) == 0,
            "passed_rules": passed,
            "failed_rules": failed,
            "details": results
        }

# Example usage
validator = CustomComplianceValidator(frameworks=["ECOA", "FCRA"])

# Test outputs from credit risk assessment
test_outputs = [
    '{"risk_level": "LOW", "explanation": "Strong credit profile with good income-to-debt ratio."}',
    '{"risk_level": "LOW", "explanation": "Strong credit profile with good income-to-debt ratio."}',
    '{"risk_level": "LOW", "explanation": "Strong credit profile with good income-to-debt ratio."}'
]

result = validator.validate(test_outputs)

print("\n📋 Compliance Validation Report")
print("=" * 60)
print(f"Compliant: {'✅ YES' if result['compliant'] else '❌ NO'}")
print(f"\nPassed rules ({len(result['passed_rules'])}):")
for rule in result['passed_rules']:
    print(f"  ✅ {rule}")
if result['failed_rules']:
    print(f"\nFailed rules ({len(result['failed_rules'])}):")
    for rule in result['failed_rules']:
        print(f"  ❌ {rule}")
```

Run it:

```bash
python custom_compliance_validator.py
```

## Step 7: Export for Regulatory Reporting

Generate compliance reports from audit trails:

**Create `generate_compliance_report.py`**:

```python
#!/usr/bin/env python3
"""
Generate compliance report from audit trails.
"""
import json
import pandas as pd
from datetime import datetime

def generate_report(trace_file: str, output_format: str = "html"):
    """Generate compliance report from JSONL audit trail."""

    # Load audit trail
    with open(trace_file) as f:
        traces = [json.loads(line) for line in f]

    # Calculate metrics
    total_runs = len(traces)
    unique_outputs = len(set(t['response_hash'] for t in traces))
    consistency = (unique_outputs == 1)
    consistency_pct = (1 / unique_outputs * 100) if unique_outputs > 0 else 100.0

    # Compliance metrics
    schema_violations = sum(not t['compliance_metrics']['schema_valid'] for t in traces)
    decision_flips = sum(t['compliance_metrics']['decision_flip'] for t in traces)
    mean_drift = sum(t['compliance_metrics']['factual_drift'] for t in traces) / total_runs

    # Generate HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Compliance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #0f62fe; }}
            .metric {{ padding: 10px; margin: 10px 0; border-left: 4px solid #0f62fe; background: #f4f4f4; }}
            .pass {{ border-left-color: #24a148; }}
            .fail {{ border-left-color: #da1e28; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #0f62fe; color: white; }}
        </style>
    </head>
    <body>
        <h1>LLM Output Drift Compliance Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Audit Trail:</strong> {trace_file}</p>

        <h2>Executive Summary</h2>
        <div class="metric {'pass' if consistency else 'fail'}">
            <strong>Consistency:</strong> {consistency_pct:.1f}% ({unique_outputs} unique output{'s' if unique_outputs != 1 else ''})
        </div>
        <div class="metric {'pass' if schema_violations == 0 else 'fail'}">
            <strong>Schema Violations:</strong> {schema_violations}
        </div>
        <div class="metric {'pass' if decision_flips == 0 else 'fail'}">
            <strong>Decision Flips:</strong> {decision_flips}
        </div>
        <div class="metric {'pass' if mean_drift < 0.05 else 'fail'}">
            <strong>Mean Drift:</strong> {mean_drift:.3f}
        </div>

        <h2>Regulatory Compliance Status</h2>
        <table>
            <tr>
                <th>Requirement</th>
                <th>Status</th>
                <th>Evidence</th>
            </tr>
            <tr>
                <td>SR 11-7 (Model Validation)</td>
                <td>{'✅ PASS' if consistency else '❌ FAIL'}</td>
                <td>Deterministic behavior: {consistency_pct:.1f}%</td>
            </tr>
            <tr>
                <td>ECOA (Consistent Decisions)</td>
                <td>{'✅ PASS' if decision_flips == 0 else '❌ FAIL'}</td>
                <td>Decision flips: {decision_flips}</td>
            </tr>
            <tr>
                <td>FSB (Output Consistency)</td>
                <td>{'✅ PASS' if mean_drift < 0.05 else '❌ FAIL'}</td>
                <td>Mean drift: {mean_drift:.3f}</td>
            </tr>
        </table>

        <h2>Model Configuration</h2>
        <pre>{json.dumps(traces[0], indent=2)[:500]}...</pre>
    </body>
    </html>
    """

    # Save report
    output_file = trace_file.replace('.jsonl', '_compliance_report.html')
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✅ Compliance report generated: {output_file}")
    return output_file

# Generate report
generate_report("traces/lab3_sql.jsonl")
```

Run it:

```bash
python generate_compliance_report.py
```

Open the HTML report in your browser to see a formatted compliance report.

## Key Takeaways

1. **Templates are JSON** - Easy to add custom tasks
2. **Modular design** - Extend components independently
3. **CI/CD ready** - Integrate into deployment pipelines
4. **Custom validators** - Implement your regulatory requirements
5. **Exportable reports** - Generate audit documentation

## Best Practices for Extensions

1. **Always test with n≥16** to detect drift
2. **Use T=0.0 and explicit seeds** for determinism
3. **Document compliance mappings** in audit trails
4. **Version your prompts** (metadata section)
5. **Validate cross-provider** before production deployment

## Quiz: Test Your Understanding

??? question "Where do you add custom tasks?"
    **Answer**: `prompts/templates.json` - add a new top-level key with task configuration.

??? question "What's the minimum number of runs recommended for drift testing?"
    **Answer**: 16 (n=16), as used in the paper's methodology.

??? question "How do you ensure determinism in custom tasks?"
    **Answer**: Set `temperature: 0.0` and `seed: 42` in the template, and test consistency with multiple runs.

## Next Steps

You've completed all workshop labs! Now you can:

1. Review [API Reference](../resources/api.md) for detailed documentation
2. Check [Troubleshooting Guide](../resources/troubleshooting.md) for common issues
3. Read the [full research paper](../resources/paper.md)
4. Contribute improvements via [GitHub](https://github.com/ibm-client-engineering/output-drift-financial-llms)

---

!!! success "Lab 6 Complete! 🎉"
    You've completed the entire workshop! You can now measure drift, classify models, validate cross-provider consistency, and extend the framework for your use cases. Thank you for participating!
