# Benchmark Tasks for Financial LLM Agents

**Purpose**: Standardized evaluation tasks for measuring determinism and faithfulness of tool-using LLM agents in financial applications.

**Paper**: [Replayable Financial Agents](https://arxiv.org/abs/2601.15322) — Accepted to ICLR 2026 FinAI Workshop

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

### Quick Run

```bash
# Run all 3 benchmarks (deterministic simulation, no LLM needed)
python econometrics/benchmarks/run_all.py

# Run a single benchmark
python econometrics/benchmarks/run_all.py --task compliance_triage --n-runs 8

# Run with actual LLM tool-calling via Ollama
python econometrics/benchmarks/run_agentic_benchmark.py --model qwen2.5:7b-instruct --n-cases 10 --n-runs 8
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
├── README.md                      # This file
├── run_all.py                     # Run all 3 benchmarks (deterministic simulation)
├── run_agentic_benchmark.py       # Agentic benchmark runner (actual LLM tool-calling)
├── compliance_triage/
│   ├── task.py                    # Task definition, tools, ground truth
│   └── data/alerts.json           # 50 test alerts
├── portfolio_constraint/
│   ├── task.py                    # Task definition, tools, ground truth
│   └── data/trades.json           # 50 proposed trades
└── dataops_exception/
    ├── task.py                    # Task definition, tools, ground truth
    └── data/exceptions.json       # 50 data exceptions
```

---

## Archived three-task results (March 2026)

The table below belongs to the earlier 4,705-run, three-task study. It is
retained for that study's lineage, not as evidence for the corrected
DFAH-Bench analysis. In particular, the portfolio fixture and its dependent
task-label matches are excluded from corrected v2.

| Model | Avg decision determinism | Historical task-label match | Benchmarks |
|-------|-------------------------:|----------------------------:|-----------:|
| Qwen 2.5 7B | 98.0% | 33.4% | 3/3 |
| Granite 3.3 | 91.1% | 42.6% | 2/3 |
| GPT-OSS 20B | 77.3% | 37.3% | 3/3 |
| Gemini 2.0 Flash | 86.0% | 49.8% | 3/3 |
| Claude Sonnet 4 | 84.0% | 38.0% | 3/3 |
| Claude Opus 4 | 71.3% | 44.2% | 3/3 |
| Gemini 2.5 Pro | 59.1% | 48.7% | 2/3 |

In that bounded study, decision determinism and the historical task-label match
were not detectably correlated (r = -0.11, p = 0.63). This descriptive result
does not identify model strategy or hidden reasoning. See
[arXiv:2601.15322](https://arxiv.org/abs/2601.15322) for its original scope.

---

## References

- [Replayable Financial Agents (arXiv:2601.15322)](https://arxiv.org/abs/2601.15322)
- `econometrics/agentic/metrics/trajectory_determinism.py`
- `econometrics/agentic/metrics/faithfulness.py`
- `econometrics/agentic/harness/stress_test_runner.py`
