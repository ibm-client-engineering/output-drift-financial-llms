# Lab 1: Understanding Output Drift

## Overview

In this lab, you'll learn what output drift is, why it matters for financial AI systems, and see real examples of non-deterministic behavior in large language models.

**Duration**: ~20 minutes

!!! note "Historical workshop scope"
    This lab preserves the original workshop experiment. Its tiers summarize
    repeatability observed in those tested runs; they are not certifications of
    correctness, safety, compliance, or production readiness. For the current
    replay-eligibility and path-agreement workflow, see [Lab 8](../lab-8/README.md).

## Learning Objectives

By the end of this lab, you will:

- Understand what output drift is and how it differs from data drift
- Learn why temperature=0.0 doesn't guarantee determinism
- See real examples of drift in financial tasks
- Understand the governance and control implications for financial services
- Know which tasks are most susceptible to drift

## What is Output Drift?

**Output drift** refers to inconsistent outputs from an LLM given identical inputs and settings. Even when temperature is set to 0.0 (supposed to be deterministic), models can produce different responses across repeated queries.

### Why Does This Happen?

Several factors contribute to output drift:

1. **Non-deterministic Operations**: GPU floating-point arithmetic, parallel processing
2. **Model Updates**: Provider-side model changes without version control
3. **Infrastructure Variability**: Load balancing, server selection
4. **Sampling Strategies**: Even at temp=0.0, implementation details vary

###  Drift vs. Data Drift

| Concept | Definition | Scope |
|---------|-----------|-------|
| **Output Drift** | Inconsistent model responses for **identical inputs** | Model behavior |
| **Data Drift** | Changes in input data distribution over time | Input data |

This workshop focuses on **output drift**—the model's internal inconsistency.

## Financial Impact: Real-World Scenarios

### Scenario 1: Loan Approval Recommendations

```
Input: "Analyze credit risk for applicant with 680 credit score, $75K income,
        20% debt-to-income ratio. Recommend approval decision."

Run 1 (temp=0.0): "APPROVE - Low risk profile"
Run 2 (temp=0.0): "DENY - Moderate risk, recommend manual review"
Run 3 (temp=0.0): "APPROVE with conditions - Reduce credit limit to $10K"
```

!!! danger "Control Risk"
    Inconsistent decisions in a credit workflow call for fairness, accuracy,
    policy, and legal review. Repeatability evidence alone does not establish a
    violation of ECOA, FCRA, or another requirement. Potential consequences of
    a poorly controlled workflow include:

    - Discrimination claims
    - Regulatory fines
    - Reputational damage
    - Loss of consumer trust

### Scenario 2: Financial Document Analysis

```
Input: SEC 10-K filing, Question: "What is the company's total debt?"

Run 1: "$2.4 billion"
Run 2: "$2.4B in long-term debt, excluding short-term obligations"
Run 3: "Total debt: $2.4 billion (page 42, footnote 7)"
```

**Issue**: All factually correct, but inconsistent formatting breaks downstream automation.

### Scenario 3: Regulatory Compliance Queries

```
Input: "Is this transaction reportable under FinCEN SAR requirements?"

Run 1: "Yes, meets threshold for suspicious activity reporting"
Run 2: "Insufficient information to determine. Request additional details."
Run 3: "No, transaction appears routine"
```

!!! warning "Review Required"
    A missed required Suspicious Activity Report (SAR) can have serious
    consequences, but these three model responses do not establish whether a
    filing was required. Reportability must be determined from the full facts
    and the applicable control process. Potential consequences can include:

    - Multi-million dollar fines
    - Criminal liability
    - License revocation

## Research Findings: By the Numbers

Our research quantified drift across multiple dimensions using 480 total runs (n=16 concurrent runs per condition):

### Overall Drift Rates (Temperature = 0.0)

| Model | Size | Consistency | Tier | Observed repeatability |
|-------|------|-------------|------|------------------------|
| **Qwen2.5-7B** | 7B | **100%** | Tier 1 | High in tested runs |
| **IBM Granite-3-8B** | 8B | **100%** | Tier 1 | High in tested runs |
| **Meta Llama-3.3-70B** | 70B | 56-100% | Tier 2 | ⚠️ Task-specific |
| **Mistral Medium** | 40B | 56-100% | Tier 2 | ⚠️ Task-specific |
| **GPT-OSS-120B** | 120B | **12.5% [CI: 3.5–36.0%]** | Tier 3 | Low in tested runs |

**Bounded finding**: in these tested configurations, the 7-20B models produced
identical outputs while the 120B configuration showed 12.5% consistency. This
does not establish a general relationship between model size and repeatability.

!!! note "Understanding Statistical Notation"
    Throughout this workshop, we report **95% Confidence Intervals (CI)** for our findings. For example, "12.5% [CI: 3.5–36.0%]" means we measured 12.5% consistency, but the true value likely falls between 3.5% and 36.0%.

    In the historical analysis, the reported Tier 1 versus Tier 3 tests had
    𝑝 < 0.0001 under their stated null model. That result does not identify the
    cause of the difference or generalize it beyond the compared configurations.

### Drift by Task Type (Temperature = 0.0)

| Task | Consistency | Observed context |
|------|-------------|------------------|
| SQL Generation | 100% | Structured output in the tested runs |
| Summarization | 100% | Narrow output format in the tested runs |
| RAG (Text-to-SQL) | 93.75% | Retrieval was part of the captured stack |
| RAG (General) | 75-87.5% | Broader natural-language outputs |

### Impact of Temperature

At **temperature = 0.2** in the workshop:

| Task | Consistency | Mean Drift | Factual Drift Range |
|------|-------------|------------|---------------------|
| RAG | 56.25% | 0.081 | 0.000 - 0.375 |
| SQL | 100% | 0.000 | 0.000 |
| Summarization | 100% | 0.000 | 0.000 |

!!! insight "Key Takeaway"
    In the tested retrieval-augmented condition, moving from temperature 0.0
    to 0.2 materially reduced exact-output agreement. This bounded comparison
    does not establish a universal temperature effect.

## Visualizing Drift

### Example: Consistency Across Model Tiers

```
Tier Classification (16 concurrent runs, temp=0.0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tier 1 (7-20B):
Qwen2.5-7B      ████████████████████  100% ✅
Granite-3-8B    ████████████████████  100% ✅
GPT-OSS-20B     ████████████████████  100% ✅

Tier 2 (40-70B):
Llama-3.3-70B   ████████████████      80%  △
Mistral Medium  ████████████████      80%  △

Tier 3 (120B+):
GPT-OSS-120B    ██▌                   12.5% ❌
```

The GPT-OSS-20B line is retained from a separate workshop check; it was not one
of the five configurations in the 480-run matrix summarized above.

### Example: Drift Heat map (Temperature Sensitivity)

```
Task Type vs. Temperature
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

             T=0.0   T=0.1   T=0.2   T=0.5
SQL          🟢100%  🟢100%  🟢100%  🟡95%
Summarize    🟢100%  🟢100%  🟢100%  🟡92%
RAG-SQL      🟢94%   🟡88%   🟡75%   🔴45%
RAG-General  🟡87%   🟡70%   🔴56%   🔴25%

🟢 Low drift  🟡 Moderate  🔴 High drift
```

## Three Types of Drift

### 1. **Syntactic Drift**
Changes in formatting, whitespace, or presentation without semantic changes.

```
Run 1: "Total Assets: $2,400,000,000"
Run 2: "Total Assets: $2.4B"
Run 3: "Total Assets: 2.4 billion USD"
```

**Impact**: Breaks parsing logic, automation fails

### 2. **Semantic Drift**
Changes in meaning or interpretation.

```
Run 1: "High risk - recommend denial"
Run 2: "Moderate risk - manual review suggested"
Run 3: "Acceptable risk with conditions"
```

**Impact**: Different business outcomes, inconsistent decisions

### 3. **Factual Drift**
Contradictory or incorrect information across runs.

```
Run 1: "Company reported $500M revenue in Q4"
Run 2: "Q4 revenue was $550M according to the filing"
Run 3: "Revenue not disclosed in available documents"
```

**Impact**: Conflicting recommendations and potential downstream control failures

## Governance Context

### Why Financial Services Care

1. **Model risk**: Teams need evidence about limitations, changes, and repeatability
2. **Fair lending**: Credit workflows require separate fairness, accuracy, and legal controls
3. **Customer communications**: Explanations and adverse-action processes depend on the use case
4. **Records and review**: Some regulated activities require records that support later reconstruction

### The Drift Challenge

Repeatability does not prove compliance, but unexplained variation can make a
system harder to validate, monitor, and reconstruct. This lab measures that
variation so the responsible reviewers can ask better questions.

## Hands-On: Observe Drift in Action

Let's see drift firsthand with a simple example:

### Step 1: Create a Test Script

Create a file called `test_drift_simple.py`:

```python
import os
from openai import OpenAI

# Use Ollama (or change to your provider)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Not actually used by Ollama
)

prompt = "What is 2+2? Answer with just the number."

print("Testing drift with 5 identical runs:\n")
for i in range(1, 6):
    response = client.chat.completions.create(
        model="qwen2.5:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        seed=42  # Explicit seed
    )
    answer = response.choices[0].message.content
    print(f"Run {i}: {answer}")
```

### Step 2: Run the Test

```bash
python test_drift_simple.py
```

### Expected Output

You'll likely see variation even for this simple task:

```
Testing drift with 5 identical runs:

Run 1: 4
Run 2: 4
Run 3: The answer is 4
Run 4: 4
Run 5: 2 + 2 = 4
```

!!! question "Discussion Point"
    Why do you think even a simple arithmetic question shows drift?

## Key Takeaways

1. **Temperature=0.0 ≠ Determinism**: Even "deterministic" settings show drift
2. **Task Matters**: Structured tasks (SQL) are more stable than open-ended tasks (RAG)
3. **Governance Relevance**: Inconsistency raises control and compliance questions in regulated workflows
4. **Provider Variance**: Different models/providers show different drift characteristics
5. **Measurement is Essential**: You can't manage what you don't measure

## Quiz: Test Your Understanding

??? question "Question 1: What is output drift?"
    **Answer**: Inconsistent outputs from an LLM given identical inputs and settings.

??? question "Question 2: Why is drift a problem for financial services?"
    **Answer**: It can make decisions harder to reproduce, explain, and govern,
    which raises control and compliance questions that require separate review.

??? question "Question 3: Which task showed the highest drift in research?"
    **Answer**: RAG (Retrieval-Augmented Generation) tasks, especially at temperature > 0.0.

??? question "Question 4: Does setting temperature=0.0 eliminate drift?"
    **Answer**: No guarantee follows from temperature alone. Some tested
    configurations reached 100% agreement at T=0.0, while another reached
    12.5%; the configurations also differed in model family and serving stack.

## Next Steps

Now that you understand what output drift is and why it matters:

1. **Proceed to [Lab 2: Setting Up Your Environment](../lab-2/README.md)** to configure API keys and providers
2. Review the full research paper in `docs/resources/paper.md`
3. Think about how drift might affect your own AI applications

## Further Reading

- [Model Risk Management (SR 11-7)](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Fair Lending and AI](https://www.consumerfinance.gov/about-us/newsroom/cfpb-acts-to-protect-the-public-from-black-box-credit-models-using-complex-algorithms/)

---

!!! success "Lab 1 Complete!"
    You now understand output drift and its implications. Ready to configure your environment? Move on to [Lab 2: Setting Up Your Environment](../lab-2/README.md)!
