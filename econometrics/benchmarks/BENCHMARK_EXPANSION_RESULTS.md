# Benchmark Expansion Results

**Date**: December 19, 2025
**Status**: All benchmarks expanded to 50 test cases

---

## Test Data Summary

| Benchmark | Test Cases | Ground Truth Distribution |
|-----------|------------|---------------------------|
| **Compliance Triage** | 50 | escalate: 15, dismiss: 25, investigate: 10 |
| **Portfolio Constraint** | 50 | approve: 25, reject: 18, modify: 7 |
| **DataOps Exception** | 50 | auto_fix: 30, escalate: 15, quarantine: 5 |

---

## Benchmark Runner Validation

Ran `run_all.py` with 10 tests, 3 runs each:

| Benchmark | Action Det. | Sig. Det. | Dec. Det. | Accuracy |
|-----------|-------------|-----------|-----------|----------|
| compliance_triage | 100% | 100% | 100% | 50% |
| portfolio_constraint | 100% | 100% | 100% | 100% |
| dataops_exception | 100% | 100% | 100% | 80% |

**Note**: 100% determinism is expected for mock/simulated agent. Real LLM evaluation will show variation.

---

## Data Files Created

```
econometrics/benchmarks/
├── compliance_triage/data/
│   └── alerts.json                    # 50 compliance alerts
├── portfolio_constraint/data/
│   └── trades.json                    # 50 proposed trades
├── dataops_exception/data/
│   └── exceptions.json                # 50 data exceptions
├── results/
│   └── benchmark_results_*.json       # Timestamped results
└── run_all.py                         # Unified benchmark runner
```

---

## Test Case Design

### Compliance Triage (50 alerts)

**Escalate cases (15)**: Clear violations requiring human review
- Sanctions hits (e.g., Shadow Corp, Belarus)
- PEP involvement (politically exposed persons)
- Structuring patterns (just under $10K threshold)
- High-risk sectors (casinos, weapons, extractive industries)
- FCPA risk (payments to government officials)

**Dismiss cases (25)**: False positives with benign explanations
- Regular intercompany transfers
- Established supplier payments
- Normal business expenses
- Low-risk countries with minor flags
- Holiday shopping patterns

**Investigate cases (10)**: Need more information
- Trust account disbursements
- Complex fund structures
- Trade documentation mismatches
- Scrap metal exports
- Charitable program funding

### Portfolio Constraint (50 trades)

**Approve cases (25)**: Within all limits
- Small positions (< 3% of portfolio)
- High liquidity stocks
- Diversifying trades
- Full position exits

**Reject cases (18)**: Clear violations
- Position limit exceeded (> 5%)
- Liquidity insufficient (> 3 days to trade)
- Cash reserve violation (< 2%)
- Penny stock illiquidity

**Modify cases (7)**: Partially compliant
- Close to position limit
- Moderate liquidity concern
- Sector concentration risk

### DataOps Exception (50 exceptions)

**Auto-fix cases (30)**: Clear corrections
- Negative values → absolute value
- Date format conversions (MM/DD/YYYY → ISO)
- Ticker mappings (GOOG → GOOGL)
- Typo corrections (NASDQ → NASDAQ)
- Bid/ask swaps

**Escalate cases (15)**: Need human judgment
- Missing CUSIPs for new securities
- Zero prices (could be trading halts)
- Invalid LEIs
- MiFID venue requirements
- Settlement timeline inconsistencies

**Quarantine cases (5)**: Data corruption
- Values 1000x out of range
- Commission > notional value
- Zero FX rates
- Private securities without exchange

---

## Next Steps

1. [x] Create 50 test cases per benchmark
2. [x] Implement unified runner
3. [ ] Run with real LLM agents (Tier 1, Tier 2, Frontier)
4. [ ] Measure actual determinism/faithfulness across model tiers
5. [ ] Generate paper tables with results

---

## Commands

```bash
# Run all benchmarks (full)
python econometrics/benchmarks/run_all.py

# Run specific benchmark
python econometrics/benchmarks/run_all.py --task compliance_triage

# Quick validation (10 tests, 3 runs)
python econometrics/benchmarks/run_all.py --max-tests 10 --n-runs 3

# Full benchmark suite
python econometrics/benchmarks/run_all.py --max-tests 50 --n-runs 10
```
