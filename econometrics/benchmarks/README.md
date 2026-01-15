# V3 Benchmark Tasks for Financial LLM Agents

**Purpose**: Standardized evaluation tasks for measuring determinism and faithfulness of tool-using LLM agents in financial applications.

**Target**: ICLR 2026 FinAI Workshop (January 30, 2026 deadline)

---

## Overview

Three benchmark tasks spanning the compliance, investment, and operations domains:

| Task | Domain | Primary Metric | Secondary Metric |
|------|--------|----------------|------------------|
| **Compliance Triage** | Regulatory | Decision Determinism | Evidence Grounding |
| **Portfolio Constraint** | Investment | Constraint Satisfaction | Position Limit Adherence |
| **DataOps Exception** | Operations | Signature Determinism | Action Determinism |

---

## 1. Compliance Triage

**Scenario**: A compliance agent receives an alert about a potentially suspicious transaction and must decide whether to escalate, dismiss, or request more information.

**Tools Available**:
- `search_precedents(query)` - Search historical compliance cases
- `get_customer_profile(id)` - Retrieve customer risk profile
- `check_sanctions_list(name)` - Screen against sanctions lists
- `calculate_risk_score(factors)` - Compute transaction risk score

**Ground Truth Labels**:
- `escalate` - Alert matches known fraud patterns
- `dismiss` - Alert is a false positive
- `investigate` - Insufficient information

**Metrics**:
- **Decision Determinism**: Does the agent make the same escalation decision across runs?
- **Evidence Grounding**: Are cited precedents actually retrieved?

**Example Input**:
```json
{
  "alert_id": "TXN-2025-001234",
  "transaction": {
    "amount": 47500,
    "currency": "USD",
    "sender": "ABC Corp",
    "receiver": "XYZ Holdings",
    "country": "Cayman Islands"
  },
  "flags": ["unusual_amount", "offshore_destination"]
}
```

**Expected Tool Sequence**:
1. `check_sanctions_list("XYZ Holdings")`
2. `get_customer_profile("ABC Corp")`
3. `search_precedents("offshore wire unusual amount")`
4. `calculate_risk_score({...})`

---

## 2. Portfolio Constraint Checking

**Scenario**: A portfolio management agent must recommend trades while respecting position limits, sector exposure caps, and regulatory constraints.

**Tools Available**:
- `get_current_holdings(portfolio_id)` - Current portfolio positions
- `get_market_data(ticker)` - Current price and volume
- `check_position_limit(ticker, quantity)` - Verify against limits
- `calculate_sector_exposure(sector)` - Current sector exposure %
- `get_regulatory_constraints(region)` - Regulatory limits

**Constraints to Check**:
- Position limits (max 5% single stock)
- Sector caps (max 25% any sector)
- Liquidity requirements (3-day volume coverage)
- Regulatory minimums (cash reserves)

**Ground Truth Labels**:
- `approve` - Trade satisfies all constraints
- `reject` - Trade violates constraints (with specific violation)
- `modify` - Trade requires adjustment (with suggested modification)

**Metrics**:
- **Constraint Satisfaction**: Does the agent correctly check all constraints?
- **Position Limit Adherence**: Are violations correctly identified?

**Example Input**:
```json
{
  "portfolio_id": "FUND-2025-ALPHA",
  "proposed_trade": {
    "action": "buy",
    "ticker": "AAPL",
    "quantity": 10000,
    "price": 195.50
  },
  "reason": "Increase tech exposure"
}
```

---

## 3. DataOps Exception Handling

**Scenario**: A data operations agent must resolve data quality exceptions in a financial data pipeline, deciding whether to auto-fix, escalate, or quarantine records.

**Tools Available**:
- `get_exception_details(exception_id)` - Full exception context
- `query_reference_data(field, value)` - Look up canonical values
- `get_historical_fixes(pattern)` - Search past resolutions
- `validate_fix(field, old_value, new_value)` - Check proposed fix
- `apply_fix(exception_id, fix)` - Apply and log fix
- `escalate_to_human(exception_id, reason)` - Escalate to team

**Exception Types**:
- Format errors (dates, numbers, identifiers)
- Reference data mismatches (ticker, CUSIP, ISIN)
- Business rule violations (negative prices, impossible dates)
- Missing required fields

**Ground Truth Labels**:
- `auto_fix` - Fix can be automatically applied
- `escalate` - Requires human judgment
- `quarantine` - Cannot proceed, needs investigation

**Metrics**:
- **Signature Determinism**: Same tools with same arguments?
- **Action Determinism**: Same tools called (any order)?

**Example Input**:
```json
{
  "exception_id": "DQ-2025-00789",
  "source": "market_data_feed",
  "field": "trade_price",
  "value": "-45.23",
  "rule_violated": "price_must_be_positive",
  "record": {
    "ticker": "MSFT",
    "date": "2025-01-15",
    "volume": 12500000
  }
}
```

---

## Evaluation Protocol

### Run Configuration

```python
# Configuration for benchmark evaluation
BENCHMARK_CONFIG = {
    "n_runs": 10,           # Runs per input
    "temperatures": [0.0],   # Temperature settings
    "models": [
        "claude-opus-4-5",
        "gemini-2.5-pro",
        "llama-3-3-70b",
        "gpt-oss-120b"
    ],
    "architectures": [
        "unconstrained",
        "schema_first",
        "policy_gated"
    ]
}
```

### Metrics Computation

```python
# Determinism metrics
action_determinism = count_matching_tool_sequences(runs) / len(runs)
signature_determinism = count_matching_signatures(runs) / len(runs)
decision_determinism = count_matching_decisions(runs) / len(runs)

# Faithfulness metrics
evidence_grounding = count_grounded_claims(decision) / total_claims(decision)
constraint_satisfaction = count_satisfied_constraints(decision) / total_constraints
```

---

## Directory Structure

```
econometrics/benchmarks/
├── README.md                    # This file
├── compliance_triage/
│   ├── __init__.py
│   ├── task.py                  # Task definition and tools
│   ├── data/                    # Test inputs
│   │   ├── alerts.json          # 50 test alerts
│   │   └── ground_truth.json    # Expected labels
│   └── evaluate.py              # Evaluation script
│
├── portfolio_constraint/
│   ├── __init__.py
│   ├── task.py
│   ├── data/
│   │   ├── trades.json          # 50 proposed trades
│   │   └── ground_truth.json
│   └── evaluate.py
│
├── dataops_exception/
│   ├── __init__.py
│   ├── task.py
│   ├── data/
│   │   ├── exceptions.json      # 50 data exceptions
│   │   └── ground_truth.json
│   └── evaluate.py
│
└── run_all.py                   # Full benchmark suite
```

---

## Expected Baseline Results

Based on V3 module validation:

| Model Tier | Task | Decision Det. | Evidence Grounding |
|------------|------|---------------|--------------------|
| Tier 1 (7-20B) | Compliance Triage | 100% | 95%+ |
| Tier 1 (7-20B) | Portfolio Constraint | 100% | 100% |
| Tier 1 (7-20B) | DataOps Exception | 100% | 90%+ |
| Frontier | Compliance Triage | 85-95% | 100% |
| Frontier | Portfolio Constraint | 95-100% | 100% |
| Frontier | DataOps Exception | 80-90% | 95%+ |
| Tier 3 (120B+) | All Tasks | 10-40% | 70-80% |

---

## Next Steps

1. [ ] Create task.py for each benchmark
2. [ ] Generate 50 test inputs per task
3. [ ] Define ground truth labels
4. [ ] Implement evaluation scripts
5. [ ] Run baseline experiments across all model tiers
6. [ ] Document results in paper

---

## References

- ICLR 2026 FinAI Workshop: Replayable Financial Agents paper
- `econometrics/agentic/metrics/trajectory_determinism.py`
- `econometrics/agentic/metrics/faithfulness.py`
- `econometrics/agentic/harness/stress_test_runner.py`
