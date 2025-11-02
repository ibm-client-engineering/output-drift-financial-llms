# Paper Changes Summary for arXiv Submission

## Date: 2025-11-01
## Paper: LLM Output Drift - ACM ICAIF 2025

---

## Priority 1: New GitHub Repository Structure ✓

**Location**: `/Users/rsk/Downloads/ai4f-drift-runner-pro/output-drift-financial-llms/`

### Repository Contents:
```
output-drift-financial-llms/
├── README.md                           # Compelling practitioner-focused guide
├── LICENSE                             # Apache 2.0
├── requirements.txt                    # Pinned dependencies
├── harness/
│   ├── __init__.py                    # Package init
│   ├── deterministic_retriever.py     # SEC 10-K structure-aware retrieval
│   ├── task_definitions.py            # RAG, SQL, Summary tasks
│   └── cross_provider_validation.py   # Multi-provider consistency gates
├── data/
│   └── generate_toy_finance.py        # Synthetic database generator (Faker)
├── prompts/
│   └── templates.json                  # Complete prompt templates from Appendix D
└── examples/
    └── sample_audit_trail.jsonl        # Bi-temporal audit trail samples
```

**Key Features**:
- Based on your actual codebase (`runner.py`, `rag_task.py`, `rag_corpus.py`, `watsonx.py`)
- Production-ready simplified versions for public release
- Complete with setup instructions for Ollama + watsonx.ai
- Finance-calibrated tolerance thresholds (±5% GAAP materiality)
- Regulatory mapping to FSB/CFTC/BIS requirements

---

## Priority 2: Paper Enhancements ✓

### Enhancement 1: New Table 1 - Tier Summary (Introduction)

**Location**: Line 716-735, after "Our key empirical finding" paragraph

**Added**:
```latex
Table~\ref{tab:tier-summary} summarizes our model tier classification...

\begin{table}[h]
\caption{Model Tiers for Financial Compliance: Deployment Decision Matrix}
\label{tab:tier-summary}
...
Tier 1: 7-8B models → 100% consistency → Full compliance
Tier 2: 40-70B models → 56-100% → Limited compliance (structured tasks only)
Tier 3: 120B models → 12.5% → Requires validation (non-compliance-critical)
\end{table}
```

**Impact**: Provides executive summary of key findings upfront for Day 1 readers

---

### Enhancement 2: Keywords Update

**Location**: Line 699

**Changed**:
```diff
- keywords: output drift, LLMs, financial services, ...
+ keywords: output drift, LLMs, financial services, ..., model-tiers, slm-finance
```

**Impact**: Better arXiv discoverability for small model and tier-based research

---

### Enhancement 3: Contribution Count Fix

**Location**: Line 750

**Changed**:
```diff
- Our experimental investigation addresses this gap with three key contributions:
+ Our experimental investigation addresses this gap with four key contributions:
```

**Impact**: Corrects mismatch (paper lists 4 bullet points)

---

### Enhancement 4: Enhanced GitHub Section

**Location**: Line 1342-1370 (before References)

**Changed**:
- Added GitHub repository URL placeholder: `https://github.com/[TBD]/output-drift-financial-llms`
- Added repository contents structure with file descriptions
- Added Quick Start code block for 5-minute reproduction
- Enhanced data source documentation
- Added cross-deployment reproducibility notes

**Impact**: Immediate practitioner value + arXiv reproducibility standards

---

## Table Numbering Cascade ✓

**Automatic LaTeX renumbering** (no manual changes needed):
- New Table 1: `tab:tier-summary` (Tier Summary - INTRODUCTION)
- Table 2: `tab:cross-provider` (Cross-Provider Multi-Model Validation)
- Table 3: `tab:baseline` (Baseline results at T=0.0)
- Table 4: `tab:cross-provider-data` (Cross-provider validation with CIs)
- Table 5: `tab:all-results` (All experimental results)
- Table 6: `tab:model-task-breakdown` (Model Performance by Task Type)
- Table 7: `tab:model-tiers` (Model Tiered Classification)
- Table 8: `tab:deployment-guide` (Model Selection Guidelines)
- Table 9: `tab:drift` (Drift patterns at T=0.2)
- Table 10: `tab:regulatory-mapping` (Finance guidance mapping)
- Table 11: `tab:prompts` (Actual prompt templates - Appendix D)

All `\ref{tab:...}` commands automatically updated by LaTeX.

---

## Compilation Status ✓

**Final output**: `output_drift.pdf`
- **Pages**: 11 (perfect for arXiv)
- **Size**: 1.37 MB
- **Errors**: None
- **Warnings**: Bibliography format warnings (acceptable)

**Commands used**:
```bash
pdflatex output_drift.tex
bibtex output_drift
pdflatex output_drift.tex
pdflatex output_drift.tex
```

---

## arXiv Submission Checklist

### Metadata:
```
Title: LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows
Authors: Raffi Khatchadourian (IBM), Rolando Franco (IBM)
Categories: cs.CL (Primary); cs.LG; q-fin.TR
Comments: 11 pages, 5 figures, 11 tables. Accepted at ACM ICAIF 2025
Keywords: output drift, Large Language Models, financial services, nondeterminism,
          regulatory compliance, cross-provider validation, reproducibility,
          model-tiers, slm-finance
```

### Quality Checks:
- [x] All tables numbered sequentially
- [x] All figures referenced correctly
- [x] GitHub URL placeholder (update when repo is live)
- [x] Page count: 11 pages ✓
- [x] No Unicode emoji characters (replaced with LaTeX symbols)
- [x] Neutral tone on all model comparisons
- [x] Bibliography compiles without errors

---

## GitHub URL - Action Required

**Current placeholder**: `https://github.com/[TBD]/output-drift-financial-llms`

**Next steps**:
1. Create GitHub repository under your account or IBM organization
2. Push `output-drift-financial-llms/` contents
3. Update paper: Replace `[TBD]` with actual GitHub username/org
4. Update README.md with final arXiv link when available

**Suggested URLs**:
- `https://github.com/raffikhatchadourian/output-drift-financial-llms`
- `https://github.com/ibm-research/output-drift-financial-llms`

---

## Day 1 Launch Strategy

### Announcement Tweet Draft:
```
📊 NEW on arXiv: Why 7B models beat 120B for regulated finance

Shocking finding: Smaller LLMs (7-8B) = 100% deterministic
Large models (120B) = 12.5% consistency at T=0! 🤯

Our framework delivers audit-ready AI today.

Paper: [arXiv link]
Code: https://github.com/[YOUR-ORG]/output-drift-financial-llms

#FinAI #LLMs #Compliance

[Attach screenshot of new Table 1]
```

### Target Downloads: 100+ on Day 1
- Compelling README ✓
- 5-minute Quick Start ✓
- Complete working code ✓
- Clear tier recommendations ✓
- Regulatory mapping ✓

---

## Files Ready for Publication

1. **Paper**: `/Users/rsk/Downloads/ai4f-drift-runner-pro/main.tex/output_drift.pdf`
2. **Repository**: `/Users/rsk/Downloads/ai4f-drift-runner-pro/output-drift-financial-llms/`
3. **LaTeX source**: `/Users/rsk/Downloads/ai4f-drift-runner-pro/main.tex/output_drift.tex`

---

## Summary of Changes

### Repository (Priority 1):
- ✅ Complete evaluation framework based on your actual code
- ✅ README with practitioner focus and model tier table
- ✅ All supporting files (requirements.txt, templates, examples, LICENSE)
- ✅ Production-ready Python harness (DeterministicRetriever, task definitions)
- ✅ Synthetic database generator with Faker
- ✅ Sample audit trails with regulatory mappings

### Paper (Priority 2):
- ✅ New Table 1: Tier Summary in Introduction (neutral tone, no emojis)
- ✅ Enhanced keywords: + model-tiers, slm-finance
- ✅ Fixed contribution count: three → four
- ✅ Enhanced GitHub section with URL placeholder and Quick Start
- ✅ All table references automatically cascaded
- ✅ Paper compiles cleanly (11 pages, no errors)

---

**Status**: READY FOR ARXIV SUBMISSION 🚀

**Remaining action**: Replace `[TBD]` with actual GitHub URL after repository creation
