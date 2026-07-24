# Lab 4: Analyzing Drift Metrics

## Overview

In this lab, you'll learn how to analyze experimental results, generate visualizations, and understand the 3-tier model classification system that emerged from the research.

**Duration**: ~25 minutes

!!! note "Historical workshop scope"
    The three tiers below are descriptive labels for the original experiment,
    not compliance, safety, or deployment certifications. Model behavior is
    configuration- and task-dependent. Requalify the exact system and inspect
    decision and tool-path evidence before operational use.

## Learning Objectives

By the end of this lab, you will:

- Understand the 3-tier model classification (Tier 1, 2, 3)
- Calculate and interpret drift metrics (consistency, Jaccard similarity)
- Generate visualizations from audit trails
- Identify high- and low-repeatability configurations in the exercise
- Use metrics to prioritize further validation

## Prerequisites

- Completed [Lab 3: Running Your First Experiment](../lab-3/README.md)
- Audit trails in `traces/` directory
- Python packages: `pandas`, `matplotlib`, `seaborn`

## The 3-Tier Model Classification

The original experiment produced three descriptive repeatability bands. The
small and large configurations also differed in model family and serving stack,
so the results do not isolate model size as a cause.

### Tier 1: High Observed Repeatability (100% Consistency @ T=0.0)

**Models**: 7-20B parameter models
- Qwen2.5-7B-Instruct (Ollama)
- IBM Granite-3-8B-Instruct (watsonx.ai)
- GPT-OSS-20B (Ollama)

**Characteristics**:
- ✅ Identical outputs in the tested T=0.0 runs
- ✅ Perfect schema compliance
- ✅ Zero decision flips
- △ Correctness and operational controls require separate validation

**Candidate follow-up tests**:
- Decision and tool-path replay on the intended task
- Argument- and result-aware capture
- Accuracy, policy, fairness, and control validation

!!! success "The Counterintuitive Finding"
    In this bounded experiment, the tested 7-20B configurations repeated more
    consistently than the tested 120B configuration. Treat that as a prompt to
    test the actual deployment configuration, not as a model-size law.

### Tier 2: Task-Specific (56-100% Consistency @ T=0.0)

**Models**: 40-70B parameter models
- Meta Llama-3.3-70B-Instruct
- Mistral Medium (2505)

**Characteristics**:
- ✅ 100% consistent for **SQL/structured tasks**
- ⚠️ 56-80% consistent for **RAG tasks**
- △ Task-dependent reliability

**Observed pattern**:
- Higher agreement on SQL and structured tasks
- Lower agreement on RAG tasks
- Task-specific qualification remains necessary

### Tier 3: Low Observed Repeatability (12.5% Consistency @ T=0.0)

**Models**: 120B+ parameter models
- GPT-OSS-120B (via watsonx.ai)

**Characteristics**:
- ❌ Only **12.5% consistent** (2/16 runs identical)
- ❌ High drift across all task types
- △ Requires investigation before any replay-dependent use

**Next step**:
- Reproduce the result under a frozen, tool-aware configuration
- Evaluate capability and controls separately from repeatability

---

## Option A: Use Built-in Analysis Tools (Quick Start)

The repository includes ready-to-run analysis tools. Use these if you want quick results:

### Generate Visualizations

```bash
# Generate drift visualizations from your experimental results
python plot_results.py traces/lab3_sql.jsonl traces/lab3_rag.jsonl
```

This creates:
- Consistency comparison charts
- Temperature sensitivity plots
- Cross-provider validation graphs

**Output**: PNG files in `results/` directory

### Generate LaTeX Tables

```bash
# Generate publication-ready tables from results
python make_tables.py results/*.csv
```

This generates LaTeX table code that you can include in reports or papers.

!!! tip "Reproducible Analysis Tools"
    These are the same tools used to generate figures and tables in the research paper. They include all statistical analysis and proper formatting.

---

## Option B: Build Your Own Analysis Scripts (Learning Path)

For deeper understanding, create custom analysis scripts:

### Step 1: Load and Analyze Audit Trails

Create `analyze_metrics.py`:

```python
import json
import pandas as pd
from collections import Counter

def load_traces(filepath):
    """Load JSONL audit trail."""
    with open(filepath) as f:
        return [json.loads(line) for line in f]

def calculate_consistency(traces):
    """Calculate consistency percentage."""
    response_hashes = [t["response_hash"] for t in traces]
    unique_hashes = set(response_hashes)
    most_common = Counter(response_hashes).most_common(1)[0]

    return {
        "total_runs": len(traces),
        "unique_responses": len(unique_hashes),
        "consistency_pct": (most_common[1] / len(traces)) * 100,
        "most_common_count": most_common[1]
    }

def calculate_drift_metrics(traces):
    """Calculate mean drift and compliance metrics."""
    factual_drifts = [t["compliance_metrics"]["factual_drift"] for t in traces]
    schema_violations = sum(not t["compliance_metrics"]["schema_valid"] for t in traces)
    decision_flips = sum(t["compliance_metrics"]["decision_flip"] for t in traces)

    return {
        "mean_drift": sum(factual_drifts) / len(factual_drifts),
        "max_drift": max(factual_drifts),
        "schema_violations": schema_violations,
        "decision_flips": decision_flips
    }

# Example usage
traces_sql = load_traces("traces/lab3_sql.jsonl")
traces_rag = load_traces("traces/lab3_rag.jsonl")

print("📊 SQL Task Analysis (T=0.0, n=16)")
print("=" * 60)
consistency_sql = calculate_consistency(traces_sql)
drift_sql = calculate_drift_metrics(traces_sql)
print(f"Consistency: {consistency_sql['consistency_pct']:.1f}%")
print(f"Unique responses: {consistency_sql['unique_responses']}")
print(f"Mean drift: {drift_sql['mean_drift']:.3f}")
print(f"Schema violations: {drift_sql['schema_violations']}")

print("\n📊 RAG Task Analysis (T=0.0, n=16)")
print("=" * 60)
consistency_rag = calculate_consistency(traces_rag)
drift_rag = calculate_drift_metrics(traces_rag)
print(f"Consistency: {consistency_rag['consistency_pct']:.1f}%")
print(f"Unique responses: {consistency_rag['unique_responses']}")
print(f"Mean drift: {drift_rag['mean_drift']:.3f}")
```

Run it:

```bash
python analyze_metrics.py
```

**Expected output:**

```
📊 SQL Task Analysis (T=0.0, n=16)
============================================================
Consistency: 100.0%
Unique responses: 1
Mean drift: 0.000
Schema violations: 0

📊 RAG Task Analysis (T=0.0, n=16)
============================================================
Consistency: 93.8%
Unique responses: 2
Mean drift: 0.012
```

### Step 2: Visualize Tier Classification

Create `visualize_tiers.py`:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data from paper (480 runs, n=16 per condition)
tier_data = pd.DataFrame({
    "Model": ["Granite-3-8B", "Qwen2.5-7B", "Llama-3.3-70B", "Mistral-Medium", "GPT-OSS-120B"],
    "Params": ["8B", "7B", "70B", "~70B", "120B"],
    "Consistency": [100.0, 100.0, 80.0, 85.0, 12.5],
    "Tier": ["Tier 1", "Tier 1", "Tier 2", "Tier 2", "Tier 3"]
})

# Set style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# Create bar chart
colors = {"Tier 1": "#2E7D32", "Tier 2": "#F57C00", "Tier 3": "#C62828"}
ax = sns.barplot(data=tier_data, x="Model", y="Consistency", hue="Tier", palette=colors, dodge=False)

# Add threshold lines
plt.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Exact agreement (100%)')
plt.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='Illustrative review line (90%)')

# Formatting
plt.title("Model Consistency @ T=0.0 (n=16): The 3-Tier Classification", fontsize=14, fontweight='bold')
plt.xlabel("Model (Parameter Count)", fontsize=12)
plt.ylabel("Consistency (%)", fontsize=12)
plt.ylim(0, 110)
plt.legend(title="Classification", loc='upper right')

# Annotate with exact values
for i, row in tier_data.iterrows():
    ax.text(i, row["Consistency"] + 2, f"{row['Consistency']:.1f}%",
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("figures/tier_classification.png", dpi=300)
print("✅ Saved: figures/tier_classification.png")
plt.show()
```

Run it:

```bash
mkdir -p figures
python visualize_tiers.py
```

**Output visualization**:

```
Consistency @ T=0.0 (n=16)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Granite-3-8B    ████████████████████  100% (Tier 1)
Qwen2.5-7B      ████████████████████  100% (Tier 1)
Llama-3.3-70B   ████████████████       80% (Tier 2)
Mistral-Medium  █████████████████      85% (Tier 2)
GPT-OSS-120B    ██▌                  12.5% (Tier 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

!!! warning "The 120B Failure"
    GPT-OSS-120B's **12.5% consistency** means only 2 out of 16 runs matched in
    this exercise. That is a strong investigation signal, not a complete
    suitability or compliance determination.

### Step 3: Temperature Sensitivity Analysis

Visualize how temperature affects different tasks:

Create `visualize_temperature.py`:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data from paper
temp_data = pd.DataFrame({
    "Task": ["SQL", "SQL", "Summarize", "Summarize", "RAG", "RAG"],
    "Temperature": [0.0, 0.2, 0.0, 0.2, 0.0, 0.2],
    "Consistency": [100.0, 100.0, 100.0, 100.0, 93.75, 56.25],
    "Mean_Drift": [0.000, 0.000, 0.000, 0.000, 0.012, 0.081]
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Consistency by temperature
sns.barplot(data=temp_data, x="Task", y="Consistency", hue="Temperature", ax=ax1, palette="viridis")
ax1.set_title("Task Consistency: T=0.0 vs T=0.2", fontsize=14, fontweight='bold')
ax1.set_ylabel("Consistency (%)", fontsize=12)
ax1.set_xlabel("Task Type", fontsize=12)
ax1.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='Illustrative review line')
ax1.legend(title="Temperature")
ax1.set_ylim(0, 110)

# Plot 2: Mean Drift
sns.barplot(data=temp_data, x="Task", y="Mean_Drift", hue="Temperature", ax=ax2, palette="rocket")
ax2.set_title("Mean Drift: T=0.0 vs T=0.2", fontsize=14, fontweight='bold')
ax2.set_ylabel("Mean Drift (Jaccard Distance)", fontsize=12)
ax2.set_xlabel("Task Type", fontsize=12)
ax2.legend(title="Temperature")

plt.tight_layout()
plt.savefig("figures/temperature_sensitivity.png", dpi=300)
print("✅ Saved: figures/temperature_sensitivity.png")
plt.show()
```

Run it:

```bash
python visualize_temperature.py
```

**Key Insight**:
- **SQL/Summarize**: Resilient to temperature (100% even at T=0.2)
- **RAG**: Highly sensitive—drops from 93.75% → 56.25% with T=0.0 → 0.2

## Step 4: Heatmap of Drift Patterns

Create a heatmap showing drift across models and tasks:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Drift data matrix (from paper)
drift_matrix = np.array([
    [0.000, 0.000, 0.012],  # Granite-3-8B
    [0.000, 0.000, 0.012],  # Qwen2.5-7B
    [0.022, 0.018, 0.035],  # Llama-3.3-70B
    [0.015, 0.012, 0.025],  # Mistral-Medium
    [0.145, 0.122, 0.187],  # GPT-OSS-120B
])

models = ["Granite-3-8B", "Qwen2.5-7B", "Llama-3.3-70B", "Mistral-Medium", "GPT-OSS-120B"]
tasks = ["SQL", "Summarize", "RAG"]

plt.figure(figsize=(10, 6))
sns.heatmap(drift_matrix, annot=True, fmt=".3f", cmap="RdYlGn_r",
            xticklabels=tasks, yticklabels=models,
            cbar_kws={'label': 'Mean Drift'}, vmin=0, vmax=0.2)

plt.title("Drift Heatmap @ T=0.0 (n=16): Model vs Task", fontsize=14, fontweight='bold')
plt.xlabel("Task Type", fontsize=12)
plt.ylabel("Model", fontsize=12)
plt.tight_layout()
plt.savefig("figures/drift_heatmap.png", dpi=300)
print("✅ Saved: figures/drift_heatmap.png")
plt.show()
```

**Interpretation**:
- 🟢 **Green (0.000-0.020)**: Low measured drift
- 🟡 **Yellow (0.020-0.050)**: Moderate measured drift
- 🔴 **Red (>0.050)**: High measured drift

## Step 5: Repeatability Scorecard

Generate a descriptive repeatability scorecard from the exercise metrics:

```python
import pandas as pd

def repeatability_scorecard(traces):
    """Summarize replay evidence from an evaluation trace."""
    consistency = calculate_consistency(traces)
    drift = calculate_drift_metrics(traces)

    # Illustrative review rules, not regulatory requirements.
    rules = {
        "Determinism": consistency["consistency_pct"] >= 95.0,
        "Low Drift": drift["mean_drift"] < 0.05,
        "Schema Compliance": drift["schema_violations"] == 0,
        "Decision Stability": drift["decision_flips"] == 0
    }

    passed = sum(rules.values())
    total = len(rules)

    return {
        "rules": rules,
        "score": f"{passed}/{total}",
        "all_checks_passed": passed == total
    }

# Test with SQL task
traces = load_traces("traces/lab3_sql.jsonl")
scorecard = repeatability_scorecard(traces)

print("\n🎯 Repeatability Scorecard: SQL Task (Qwen2.5-7B, T=0.0)")
print("=" * 60)
for rule, passed in scorecard["rules"].items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{rule:25s}: {status}")
print(f"\nOverall Score: {scorecard['score']}")
print(f"All checks passed: {'✅ YES' if scorecard['all_checks_passed'] else '❌ NO'}")
```

**Expected output:**

```
🎯 Repeatability Scorecard: SQL Task (Qwen2.5-7B, T=0.0)
============================================================
Determinism              : ✅ PASS
Low Drift                : ✅ PASS
Schema Compliance        : ✅ PASS
Decision Stability       : ✅ PASS

Overall Score: 4/4
All checks passed: ✅ YES
```

## Validation-Priority Matrix

Use the historical metrics to decide where further validation is most urgent:

| Model | Tier | SQL | Summarize | RAG | Follow-up priority | Notes |
|-------|------|-----|-----------|-----|--------------------|-------|
| **Granite-3-8B** | 1 | ✅ | ✅ | ✅ | Path-aware replay | Exact output agreement in tested runs |
| **Qwen2.5-7B** | 1 | ✅ | ✅ | ✅ | Path-aware replay | Exact output agreement in tested runs |
| **Llama-3.3-70B** | 2 | ✅ | ✅ | ⚠️ | RAG investigation | RAG agreement was lower |
| **Mistral-Medium** | 2 | ✅ | ✅ | ⚠️ | RAG investigation | RAG agreement was lower |
| **GPT-OSS-120B** | 3 | ❌ | ❌ | ❌ | Full requalification | Low agreement in tested runs |

**Illustrative triage helper**:

```python
def validation_priority(task_type, observed_tier):
    """Return a follow-up test priority, not a deployment decision."""
    if observed_tier == 3:
        return "full requalification"
    if task_type == "rag" and observed_tier == 2:
        return "high: inspect retrieval and path variation"
    return "standard: run the frozen, path-aware suite"

# Examples
print(validation_priority("sql", observed_tier=1))
# Output: "standard: run the frozen, path-aware suite"

print(validation_priority("rag", observed_tier=2))
# Output: "high: inspect retrieval and path variation"
```

## Key Takeaways

1. **Bounded Size Pattern**: The smaller tested configurations repeated more consistently; the experiment does not isolate model size
2. **Tier 1 = Observed Agreement**: It is a descriptive repeatability label, not a certification
3. **Task Structure Matters**: SQL > Summarize > RAG for determinism
4. **Temperature is Critical**: Even T=0.2 can double drift rates
5. **Metrics Guide Investigation**: Use agreement and drift measures to target deeper validation

## Quiz: Test Your Understanding

??? question "Why are 7-20B models Tier 1 while 120B models are Tier 3?"
    **Answer**: Those labels summarize the observed workshop runs. The experiment
    does not establish why the configurations differed or support a general
    causal claim about parameter count.

??? question "Does a consistency score define regulatory compliance?"
    **Answer**: No. The exercise uses descriptive repeatability bands. A
    compliance determination requires task-specific legal, policy, accuracy,
    fairness, safety, and control review.

??? question "Which task type is most resilient to temperature increases?"
    **Answer**: SQL generation—maintains 100% consistency even at T=0.2 due to structured output format.

??? question "What does a mean drift of 0.081 indicate?"
    **Answer**: Token-set variation across runs. The Jaccard score alone cannot
    determine whether the difference is semantic, factual, or material.

## Next Steps

Now that you understand drift metrics and classification:

1. **Proceed to [Lab 5: Cross-Provider Testing](../lab-5/README.md)** to validate consistency across providers
2. Generate custom visualizations from your experimental data
3. Review the full paper metrics in `docs/resources/paper.md`

---

!!! success "Lab 4 Complete!"
    You can now analyze drift metrics and use them to prioritize follow-up
    validation. Ready for cross-provider replay? Move on to
    [Lab 5: Cross-Provider Testing](../lab-5/README.md)!
