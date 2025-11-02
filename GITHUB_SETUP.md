# GitHub Repository Setup

## Quick Push to IBM Client Engineering

```bash
cd /Users/rsk/Downloads/ai4f-drift-runner-pro/output-drift-financial-llms

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial release: LLM Output Drift evaluation framework

- Complete deterministic evaluation harness
- DeterministicRetriever with SEC 10-K structure awareness
- Cross-provider validation (Ollama + IBM watsonx.ai)
- Finance-calibrated tolerance thresholds (±5% GAAP)
- Sample audit trails with regulatory mappings
- Synthetic database generator with Faker
- Complete prompt templates from ACM ICAIF 2025 paper

Based on research published in ACM ICAIF 2025:
'LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows'
by Raffi Khatchadourian and Rolando Franco (IBM)"

# Add remote repository
git remote add origin http://github.com/ibm-client-engineering/output-drift-financial-llms.git

# Push to main branch
git branch -M main
git push -u origin main
```

## After Push

1. **Verify repository is live**: Visit http://github.com/ibm-client-engineering/output-drift-financial-llms
2. **Update paper**: The GitHub URL is already set in `output_drift.tex` line 1348
3. **Paper is ready**: No additional LaTeX changes needed - already compiled with correct URL

## Repository Structure

```
output-drift-financial-llms/
├── README.md                    # Main documentation
├── LICENSE                      # Apache 2.0
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── PAPER_CHANGES.md            # Documentation of paper enhancements
├── GITHUB_SETUP.md             # This file
├── harness/
│   ├── __init__.py
│   ├── deterministic_retriever.py
│   ├── task_definitions.py
│   └── cross_provider_validation.py
├── data/
│   └── generate_toy_finance.py
├── prompts/
│   └── templates.json
└── examples/
    └── sample_audit_trail.jsonl
```

## Next Steps for arXiv

1. **Push repository** (commands above)
2. **Verify URL works**: http://github.com/ibm-client-engineering/output-drift-financial-llms
3. **Submit to arXiv**:
   - Upload: `main.tex/output_drift.pdf`
   - Categories: `cs.CL` (Primary); `cs.LG`; `q-fin.TR`
   - Comments: "11 pages, 5 figures, 11 tables. Accepted at ACM ICAIF 2025"
4. **Day 1 announcement**: See PAPER_CHANGES.md for tweet template

## Paper Status

✅ All formatting fixes applied:
- Table 1 sizing fixed (resizebox)
- Filename wrapping fixed (\linebreak)
- GitHub URL updated to ibm-client-engineering
- Removed unnecessary "critical" language
- CCS concepts formatting (automatic via ACM template)

✅ Paper compiles cleanly:
- **Output**: output_drift.pdf
- **Pages**: 11
- **Size**: 1.37 MB
- **Errors**: None

✅ Ready for submission to arXiv and GitHub
