# Lab 6: Extending the Framework

## Overview

In this lab, you'll learn how to customize the framework for your own use cases by adding new tasks, modifying prompt templates, and integrating with your workflows.

**Duration**: ~30 minutes

## Learning Objectives

By the end of this lab, you will:

- Add custom tasks to `run_evaluation.py`
- Modify existing prompts for your domain
- Integrate the framework into CI/CD pipelines
- Create custom control-rule scaffolds
- Export results for evaluation and review

## Prerequisites

- Completed [Lab 5: Cross-Provider Testing](../lab-5/README.md)
- Understanding of JSON structure
- Familiarity with your organization's approved control requirements

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

## Step 1: Understanding Task Structure

Task prompts are defined in `run_evaluation.py` (see the `build_prompts()` function) and task formatting/validation lives in `harness/task_definitions.py`.

```bash
# View the prompt builder
grep -A 30 "def build_prompts" run_evaluation.py
```

**Current tasks:**
- `rag`: RAG Q&A over SEC 10-K filings
- `summary`: JSON summarization with schema validation
- `sql`: Text-to-SQL generation

Each task uses:
- **Task formatting functions** in `harness/task_definitions.py`
- **System prompts** built into the formatting functions
- **temperature**: 0.0 to minimize configured sampling (not a determinism guarantee)
- **seed**: 42 for reproducibility

## Step 2: Add a Custom Task - Credit Risk Analysis

Let's add a new task for credit risk assessment:

To add a custom task, define a new task configuration. Here's an example credit risk assessment task structure:

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
        "review_references": ["ECOA", "FCRA"]
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
        "review_references": ["ECOA", "FCRA"]
      }
    ],
    "system_prompt": "Classify risk as LOW, MEDIUM, or HIGH from the supplied synthetic profile. Provide a brief explanation in one sentence. The repeated-run evaluation will measure whether identical inputs produce identical outputs.",
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
from collections import Counter

from openai import OpenAI

# Define custom task configuration inline
credit_risk_task = {
    "system_prompt": "Classify risk as LOW, MEDIUM, or HIGH from the supplied synthetic profile. Provide a brief explanation in one sentence.",
    "question": "Classify credit risk (LOW/MEDIUM/HIGH) and explain in one sentence.",
    "temperature": 0.0,
    "seed": 42
}

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def run_credit_risk_assessment(profile: dict, model: str = "qwen2.5:7b-instruct", n_runs: int = 5):
    """Run credit risk assessment n times to test consistency."""
    prompt = f"""Profile:
- Credit Score: {profile['credit_score']}
- Annual Income: ${profile['income']:,}
- Debt-to-Income Ratio: {profile['debt_to_income']:.0%}
- Employment Years: {profile['employment_years']}

{credit_risk_task['question']}"""

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

    # Calculate modal exact-output agreement.
    counts = Counter(results)
    unique = len(counts)
    modal_count = counts.most_common(1)[0][1]
    consistency = modal_count / len(results) * 100

    print(f"\n📊 Results:")
    print(f"  Total runs: {n_runs}")
    print(f"  Unique outputs: {unique}")
    print(f"  Consistency: {consistency:.0f}%")
    print(f"  Status: {'✅ Exact replay agreement' if consistency == 100 else '⚠️ Drift detected'}")

    return results

# Test with a synthetic profile
profile1 = {
    "credit_score": 680,
    "income": 75000,
    "debt_to_income": 0.20,
    "employment_years": 5,
}
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
  Status: ✅ Exact replay agreement
```

## Step 4: Add Domain-Specific Documents

To use RAG with your own documents:

1. **Add documents to `data/sec/`** (or create a new folder):

```bash
mkdir -p data/custom_docs
```

2. **Build the retriever from your text files**:

```python
from pathlib import Path

from harness.deterministic_retriever import DeterministicRetriever

document_paths = sorted(Path("data/custom_docs").glob("*.txt"))
documents = [
    {
        "text": path.read_text(encoding="utf-8"),
        "source": path.stem,
        "meta": {"filepath": str(path)},
    }
    for path in document_paths
]

retriever = DeterministicRetriever(
    docs=documents,
    chunk_size=512,
    overlap=50
)
```

3. **Test retrieval**:

```python
query = "What is our company's annual revenue?"
results = retriever.retrieve(query, k=5)
for i, (snippet_id, text, metadata) in enumerate(results, 1):
    print(f"Chunk {i} ({snippet_id}): {text[:100]}...")
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
            --providers watsonx \
            --models ibm/granite-3-8b-instruct \
            --temperatures 0.0 \
            --concurrency 16 \
            --tasks sql \
            --repeats 16 \
            --output traces/ci_test

      - name: Validate consistency
        run: |
          python - <<'PY'
          from collections import defaultdict
          import json
          from pathlib import Path

          trace_files = list(Path("traces/ci_test").glob("trace_*.jsonl"))
          assert trace_files, "No trace files were produced"

          groups = defaultdict(list)
          for trace_file in trace_files:
              with trace_file.open() as handle:
                  for line in handle:
                      record = json.loads(line)
                      groups[(record["task"], record["prompt_id"])].append(
                          record["output"]
                      )

          varied = {
              group: len(set(outputs))
              for group, outputs in groups.items()
              if len(set(outputs)) != 1
          }
          assert not varied, f"Exact-output variation detected: {varied}"
          print("✅ All captured replay groups had exact output agreement")
          PY

      - name: Upload replay records
        uses: actions/upload-artifact@v4
        with:
          name: drift-test-results
          path: traces/ci_test/
```

This pipeline:
- Runs on every PR and daily
- Runs 16 repeats for each built-in SQL prompt
- Applies an intentionally strict exact-output replay assertion
- Uploads replay records as artifacts

## Step 6: Custom Control-Rule Scaffold

Create a small rule scaffold for your workflow. The keyword checks below are
teaching examples only. They are not legal interpretations, fairness tests, or
evidence of regulatory compliance.

**Create `custom_control_validator.py`**:

```python
#!/usr/bin/env python3
"""Illustrative workflow-rule checks; not a compliance determination."""
import json
from typing import Dict, List

class CustomControlValidator:
    """
    Evaluate illustrative workflow rules against captured outputs.
    """

    def __init__(self, frameworks: List[str]):
        """
        Initialize validator.

        Args:
            frameworks: Labels used to select example rules for this lab.
        """
        self.frameworks = frameworks
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, callable]:
        """Load validation rules for each framework."""
        rules = {}

        # Illustrative checks grouped under familiar labels. These simple
        # heuristics do not implement or interpret the cited laws.
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
        """Check exact agreement across the supplied examples."""
        unique_outputs = len(set(outputs))
        return unique_outputs == 1  # 100% consistency required

    def _check_no_discrimination(self, output: str) -> bool:
        """Flag a small demonstration vocabulary for manual review."""
        protected_terms = ["race", "gender", "age", "religion", "nationality"]
        return not any(term in output.lower() for term in protected_terms)

    def _check_explainability(self, output: str) -> bool:
        """Check for an explanation marker in this teaching example."""
        return "explanation" in output.lower() or "because" in output.lower()

    def _check_data_minimization(self, output: str) -> bool:
        """Flag a small demonstration vocabulary for manual review."""
        pii_indicators = ["ssn", "social security", "passport", "driver license"]
        return not any(indicator in output.lower() for indicator in pii_indicators)

    def validate(self, outputs: List[str]) -> Dict[str, any]:
        """
        Run all validation rules.

        Args:
            outputs: List of LLM outputs to validate

        Returns:
            {
                "all_example_rules_passed": bool,
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
            "all_example_rules_passed": len(failed) == 0,
            "passed_rules": passed,
            "failed_rules": failed,
            "details": results
        }

# Example usage
validator = CustomControlValidator(frameworks=["ECOA", "FCRA"])

# Test outputs from credit risk assessment
test_outputs = [
    '{"risk_level": "LOW", "explanation": "Strong credit profile with good income-to-debt ratio."}',
    '{"risk_level": "LOW", "explanation": "Strong credit profile with good income-to-debt ratio."}',
    '{"risk_level": "LOW", "explanation": "Strong credit profile with good income-to-debt ratio."}'
]

result = validator.validate(test_outputs)

print("\n📋 Example Control-Rule Report")
print("=" * 60)
print(
    "All example rules passed: "
    f"{'✅ YES' if result['all_example_rules_passed'] else '❌ NO'}"
)
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
python custom_control_validator.py
```

## Step 7: Export an Evaluation Report

Generate a review report from replay records:

**Create `generate_evaluation_report.py`**:

```python
#!/usr/bin/env python3
"""
Generate an evaluation report from replay records.
"""
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from rapidfuzz.distance import Levenshtein

def generate_report(trace_file: str, output_format: str = "html"):
    """Generate a descriptive replay report from a JSONL trace."""

    # Load replay records
    with open(trace_file) as f:
        traces = [json.loads(line) for line in f]

    # Calculate metrics
    if not traces:
        raise ValueError("Trace file contains no records")

    total_runs = len(traces)
    groups = defaultdict(list)
    for trace in traces:
        groups[(trace["task"], trace["prompt_id"])].append(trace)

    modal_count = sum(
        Counter(record["output"] for record in group).most_common(1)[0][1]
        for group in groups.values()
    )
    consistency_pct = modal_count / total_runs * 100
    consistency = all(
        len({record["output"] for record in group}) == 1
        for group in groups.values()
    )

    # Descriptive review metrics
    schema_violations = sum(
        bool(trace.get("schema_violation", False)) for trace in traces
    )
    decision_flips = 0
    distances = []
    for group in groups.values():
        reference_output = group[0]["output"]
        reference_decision = group[0].get("decision_ok")
        for trace in group:
            distances.append(
                Levenshtein.normalized_distance(
                    reference_output, trace["output"]
                )
            )
            if (
                reference_decision is not None
                and trace.get("decision_ok") != reference_decision
            ):
                decision_flips += 1
    mean_drift = sum(distances) / len(distances)

    # Generate HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Replay Evaluation Report</title>
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
        <h1>LLM Replay Evaluation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Replay Record:</strong> {trace_file}</p>

        <h2>Executive Summary</h2>
        <div class="metric {'pass' if consistency else 'fail'}">
            <strong>Modal exact-output agreement:</strong> {consistency_pct:.1f}%
            across {len(groups)} replay group{'s' if len(groups) != 1 else ''}
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

        <h2>Illustrative Review Signals</h2>
        <table>
            <tr>
                <th>Signal</th>
                <th>Status</th>
                <th>Evidence</th>
            </tr>
            <tr>
                <td>Exact output replay</td>
                <td>{'✅ PASS' if consistency else '❌ FAIL'}</td>
                <td>Observed agreement: {consistency_pct:.1f}%</td>
            </tr>
            <tr>
                <td>No decision flips in this trace</td>
                <td>{'✅ PASS' if decision_flips == 0 else '❌ FAIL'}</td>
                <td>Decision flips: {decision_flips}</td>
            </tr>
            <tr>
                <td>Illustrative normalized string-distance review line</td>
                <td>{'✅ PASS' if mean_drift < 0.05 else '❌ FAIL'}</td>
                <td>Mean drift: {mean_drift:.3f}</td>
            </tr>
        </table>

        <p><strong>Interpretation:</strong> These signals summarize the captured
        run. They do not determine correctness, safety, fairness, or regulatory
        compliance.</p>

        <h2>Model Configuration</h2>
        <pre>{json.dumps(traces[0], indent=2)[:500]}...</pre>
    </body>
    </html>
    """

    # Save report
    output_file = trace_file.replace('.jsonl', '_evaluation_report.html')
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✅ Evaluation report generated: {output_file}")
    return output_file

# Generate a report from the isolated Lab 3 run
trace_file = next(Path("traces/lab3_multi").glob("trace_*.jsonl"))
generate_report(str(trace_file))
```

Run it:

```bash
python generate_evaluation_report.py
```

Open the HTML report in your browser to see the formatted evaluation summary.

## Key Takeaways

1. **Templates are JSON** - Easy to add custom tasks
2. **Modular design** - Extend components independently
3. **CI integration** - Run an explicitly chosen replay assertion in an existing pipeline
4. **Custom rule scaffolds** - Encode organization-approved review rules
5. **Exportable reports** - Generate descriptive replay documentation

## Best Practices for Extensions

1. **Choose replay count from the decision risk and precision needed**; n=16 is the workshop example, not a universal minimum
2. **Record T=0.0 and explicit seeds when supported**, but do not treat them as determinism guarantees
3. **Document control references as review metadata**, not automated compliance findings
4. **Version your prompts** (metadata section)
5. **Requalify the intended provider stack** before relying on replay equivalence

## Quiz: Test Your Understanding

??? question "Where do you add custom tasks?"
    **Answer**: `run_evaluation.py` - add a new top-level key with task configuration.

??? question "What's the minimum number of runs recommended for drift testing?"
    **Answer**: There is no universal minimum. This workshop uses n=16; a real
    workflow should select replay count from risk, expected variation, cost,
    and the precision needed for the decision.

??? question "How do you ensure determinism in custom tasks?"
    **Answer**: You cannot ensure it from temperature and seed alone. Pin the
    full configuration, capture required channels, and measure repeatability
    across comparable runs.

## Next Steps

You've completed all workshop labs! Now you can:

1. Review [API Reference](../resources/api.md) for detailed documentation
2. Check [Troubleshooting Guide](../resources/troubleshooting.md) for common issues
3. Read the [full research paper](../resources/paper.md)
4. Contribute improvements via [GitHub](https://github.com/ibm-client-engineering/output-drift-financial-llms)

---

!!! success "Lab 6 Complete! 🎉"
    You've completed the entire workshop! You can now measure drift, classify models, validate cross-provider consistency, and extend the framework for your use cases. Thank you for participating!
