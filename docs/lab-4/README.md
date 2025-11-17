# Lab 4: Analyzing Drift Metrics

## Overview

In this lab, you'll learn how to analyze experimental results, generate visualizations, and understand the 3-tier model classification system that emerged from the research.

**Duration**: ~25 minutes

## Learning Objectives

By the end of this lab, you will:

- Understand the 3-tier model classification (Tier 1, 2, 3)
- Calculate and interpret drift metrics (consistency, Jaccard similarity)
- Generate visualizations from audit trails
- Identify compliance-safe vs non-compliant configurations
- Make deployment recommendations based on metrics

## Prerequisites

- Completed [Lab 3: Running Your First Experiment](../lab-3/README.md)
- Audit trails in `traces/` directory
- Python packages: `pandas`, `matplotlib`, `seaborn`

## The 3-Tier Model Classification

Our research revealed that **model size inversely correlates with deterministic behavior**—smaller models are more reliable for compliance!

### Tier 1: Audit-Ready (100% Consistency @ T=0.0)

**Models**: 7-8B parameter models
- Qwen2.5-7B-Instruct (Ollama)
- IBM Granite-3-8B-Instruct (watsonx.ai)

**Characteristics**:
- ✅ **100% deterministic** at T=0.0
- ✅ Perfect schema compliance
- ✅ Zero decision flips
- ✅ Audit-ready for all regulated tasks

**Recommended Use**:
- Credit decisions
- Regulatory reporting
- Client communications
- Any compliance-critical workflow

!!! success "The Counterintuitive Finding"
    **Smaller ≠ Worse**: 7-8B models achieve perfect determinism while 120B models fail—a fundamental challenge to "bigger is better" assumptions!

### Tier 2: Task-Specific (56-100% Consistency @ T=0.0)

**Models**: 40-70B parameter models
- Meta Llama-3.3-70B-Instruct
- Mistral Medium (2505)

**Characteristics**:
- ✅ 100% consistent for **SQL/structured tasks**
- ⚠️ 56-80% consistent for **RAG tasks**
- △ Task-dependent reliability

**Recommended Use**:
- SQL generation only
- Structured data extraction
- **Avoid**: RAG, open-ended Q&A, retrieval tasks

### Tier 3: Non-Compliant (12.5% Consistency @ T=0.0)

**Models**: 120B+ parameter models
- GPT-OSS-120B (via watsonx.ai)

**Characteristics**:
- ❌ Only **12.5% consistent** (2/16 runs identical)
- ❌ High drift across all task types
- ❌ Unsuitable for regulated applications

**Recommendation**:
- **Do not use** for financial compliance workflows
- High-scale models trade determinism for capability

---

## Option A: Use Built-in Analysis Tools (Quick Start)

The repository includes production-ready analysis tools. Use these if you want quick results:

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

!!! tip "Production-Ready Tools"
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
plt.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Audit-Ready (100%)')
plt.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='Compliance Threshold (90%)')

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
    GPT-OSS-120B's **12.5% consistency** means only 2 out of 16 runs matched—completely unsuitable for audit trails or regulated decisions.

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
ax1.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='Compliance Threshold')
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
- 🟢 **Green (0.000-0.020)**: Compliance-safe
- 🟡 **Yellow (0.020-0.050)**: Monitor closely
- 🔴 **Red (>0.050)**: Non-compliant

## Step 5: Compliance Scorecard

Generate a compliance scorecard based on metrics:

```python
import pandas as pd

def compliance_scorecard(traces):
    """Generate compliance scorecard from audit trail."""
    consistency = calculate_consistency(traces)
    drift = calculate_drift_metrics(traces)

    # Scoring rules (from regulatory requirements)
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
        "compliant": passed == total
    }

# Test with SQL task
traces = load_traces("traces/lab3_sql.jsonl")
scorecard = compliance_scorecard(traces)

print("\n🎯 Compliance Scorecard: SQL Task (Qwen2.5-7B, T=0.0)")
print("=" * 60)
for rule, passed in scorecard["rules"].items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{rule:25s}: {status}")
print(f"\nOverall Score: {scorecard['score']}")
print(f"Compliant: {'✅ YES' if scorecard['compliant'] else '❌ NO'}")
```

**Expected output:**

```
🎯 Compliance Scorecard: SQL Task (Qwen2.5-7B, T=0.0)
============================================================
Determinism              : ✅ PASS
Low Drift                : ✅ PASS
Schema Compliance        : ✅ PASS
Decision Stability       : ✅ PASS

Overall Score: 4/4
Compliant: ✅ YES
```

## Deployment Decision Matrix

Based on metrics, here's a decision matrix for production:

| Model | Tier | SQL | Summarize | RAG | Compliance Use | Notes |
|-------|------|-----|-----------|-----|----------------|-------|
| **Granite-3-8B** | 1 | ✅ | ✅ | ✅ | **All tasks** | 100% deterministic |
| **Qwen2.5-7B** | 1 | ✅ | ✅ | ✅ | **All tasks** | 100% deterministic |
| **Llama-3.3-70B** | 2 | ✅ | ✅ | ⚠️ | SQL only | RAG drift too high |
| **Mistral-Medium** | 2 | ✅ | ✅ | ⚠️ | SQL only | RAG inconsistent |
| **GPT-OSS-120B** | 3 | ❌ | ❌ | ❌ | **None** | Non-compliant |

**Recommendation Algorithm**:

```python
def recommend_model(task_type, compliance_required):
    """Recommend model based on task and compliance needs."""
    if compliance_required:
        if task_type in ["sql", "summarize", "rag"]:
            return "Tier 1 (Granite-3-8B or Qwen2.5-7B)"
        else:
            return "Tier 1 only - evaluate before deployment"
    else:
        # Non-compliance use cases
        if task_type in ["sql", "summarize"]:
            return "Tier 1 or Tier 2"
        elif task_type == "rag":
            return "Tier 1 (Tier 2 shows drift)"
        else:
            return "Evaluate experimentally"

# Examples
print(recommend_model("sql", compliance_required=True))
# Output: "Tier 1 (Granite-3-8B or Qwen2.5-7B)"

print(recommend_model("rag", compliance_required=False))
# Output: "Tier 1 (Tier 2 shows drift)"
```

## Key Takeaways

1. **Size Paradox**: 7-8B models outperform 120B models for deterministic tasks
2. **Tier 1 = Audit-Ready**: Only 100% consistent models are compliance-safe
3. **Task Structure Matters**: SQL > Summarize > RAG for determinism
4. **Temperature is Critical**: Even T=0.2 can double drift rates
5. **Metrics Drive Decisions**: Use consistency, drift, and compliance scores to guide deployment

## Quiz: Test Your Understanding

??? question "Why are 7-8B models Tier 1 while 120B models are Tier 3?"
    **Answer**: Smaller models achieve 100% determinism through simpler architectures and less non-deterministic parallelization, while larger models trade consistency for capability.

??? question "What consistency threshold defines 'compliant' for regulated financial applications?"
    **Answer**: ≥95% consistency (our research uses 100% as the gold standard for Tier 1).

??? question "Which task type is most resilient to temperature increases?"
    **Answer**: SQL generation—maintains 100% consistency even at T=0.2 due to structured output format.

??? question "What does a mean drift of 0.081 indicate?"
    **Answer**: Moderate semantic variation across runs—approaching the threshold where factual inconsistencies emerge (>0.1).

## Next Steps

Now that you understand drift metrics and classification:

1. **Proceed to [Lab 5: Cross-Provider Testing](../lab-5/README.md)** to validate consistency across providers
2. Generate custom visualizations from your experimental data
3. Review the full paper metrics in `docs/resources/paper.md`

---

!!! success "Lab 4 Complete!"
    You can now analyze drift metrics, classify models, and make compliance-informed deployment decisions! Ready for cross-provider validation? Move on to [Lab 5: Cross-Provider Testing](../lab-5/README.md)!
