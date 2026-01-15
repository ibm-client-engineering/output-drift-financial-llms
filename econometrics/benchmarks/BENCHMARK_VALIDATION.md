# V3 Benchmark Validation Results

**Date**: December 19, 2025
**Status**: All 3 benchmarks validated and working

---

## Summary

| Benchmark | Status | Test Cases | Tools |
|-----------|--------|------------|-------|
| Compliance Triage | ✅ PASS | 3 | 4 |
| Portfolio Constraint | ✅ PASS | 3 | 5 |
| DataOps Exception | ✅ PASS | 3 | 6 |

---

## 1. Compliance Triage

**File**: `econometrics/benchmarks/compliance_triage/task.py`

**Test Results**:

| Alert ID | Amount | Flags | Ground Truth | Tools Called |
|----------|--------|-------|--------------|--------------|
| TXN-2025-001234 | $47,500 | unusual_amount, offshore | **dismiss** | sanctions, profile, precedents, risk |
| TXN-2025-001235 | $125,000 | new_customer, high_risk_country | **escalate** | sanctions (HIT!), profile, precedents, risk |
| TXN-2025-001236 | $5,000 | round_amount | **dismiss** | sanctions, profile, precedents, risk |

**Tool Execution**:
```
Alert 001234: Low risk (0.25), no sanctions hit -> DISMISS
Alert 001235: High risk (1.0), sanctions HIT, new customer -> ESCALATE
Alert 001236: Low risk (0.20), minor flag -> DISMISS
```

---

## 2. Portfolio Constraint

**File**: `econometrics/benchmarks/portfolio_constraint/task.py`

**Test Results**:

| Trade ID | Action | Notional | Violation | Ground Truth |
|----------|--------|----------|-----------|--------------|
| TRADE-2025-001 | BUY AAPL | $195K | None | **approve** |
| TRADE-2025-002 | BUY NVDA | $960K | Position + Sector | **reject** |
| TRADE-2025-003 | BUY SMALL_CAP | $2.5M | Position + Sector + Liquidity | **reject** |

**Constraint Violations Detected**:
```
Trade 001: Position 4.88% (OK), Sector would be 67.8% (already over but trade is small)
Trade 002: Position 24% (VIOLATED), Sector 86.9% (VIOLATED)
Trade 003: Position 62.5% (VIOLATED), Sector 125.4% (VIOLATED)
```

---

## 3. DataOps Exception

**File**: `econometrics/benchmarks/dataops_exception/task.py`

**Test Results**:

| Exception ID | Type | Field | Issue | Ground Truth | Fix |
|--------------|------|-------|-------|--------------|-----|
| DQ-2025-00789 | business_rule | trade_price | -45.23 | **auto_fix** | abs(-45.23) = 45.23 |
| DQ-2025-00790 | format_error | trade_date | 01/15/2025 | **auto_fix** | -> 2025-01-15 |
| DQ-2025-00791 | missing_field | cusip | None | **escalate** | Can't auto-fill |

**Tool Execution**:
```
Exception 00789: Historical fix pattern found (95% success), validate abs() -> PASS
Exception 00790: Historical fix pattern found (100% success), validate ISO date -> PASS
Exception 00791: Reference lookup failed, no canonical value -> ESCALATE
```

---

## Validation Commands

```bash
# Run all benchmarks
python econometrics/benchmarks/compliance_triage/task.py
python econometrics/benchmarks/portfolio_constraint/task.py
python econometrics/benchmarks/dataops_exception/task.py
```

---

## Next Steps for Production

1. [ ] Expand to 50 test cases per benchmark
2. [ ] Add JSON data files for inputs
3. [ ] Create ground truth annotation files
4. [ ] Implement evaluation scripts with metric computation
5. [ ] Run baseline experiments across model tiers
6. [ ] Document results in ICLR paper tables

---

## Paper Integration

These benchmarks support Section 5 (Experiments) of the ICLR 2026 paper:

- **Table 3**: Benchmark task descriptions
- **Table 4**: Baseline results by model tier
- **Figure 2**: Determinism-faithfulness frontier per task

Reference: `econometrics/paper/latex/replayable_agents.tex`
