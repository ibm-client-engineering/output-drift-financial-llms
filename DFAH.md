# DFAH: Determinism-Faithfulness Assurance Harness

A harness for measuring whether LLM agents produce consistent, auditable behavior when given the same input multiple times.

From the paper: [Replayable Financial Agents](https://arxiv.org/abs/2601.15322) (ICLR 2026 FinAI Workshop).

> This page documents the earlier research API. For the supported prospective
> package, use [`README_DFAH.md`](README_DFAH.md) and
> `dfah check-agent --agent dfah.demo:toy_agent`.

---

## Who This Is For

- **AI/ML Engineers** deploying LLM agents in production and needing to measure behavioral consistency
- **Compliance & Risk Teams** reviewing replay and execution evidence
- **Researchers** studying LLM non-determinism in tool-using agents

## What DFAH Measures

DFAH runs an agent N times on the same input and compares the trajectories:

| Metric | What It Measures |
|--------|-----------------|
| **Action Determinism** | Did the agent call the same tools? (names only) |
| **Signature Determinism** | Did it call them with the same arguments? |
| **Decision Determinism** | Did it reach the same final decision? |
| **Accuracy** | Did the decision match ground truth? |

**Earlier-study result**: Decision determinism and historical task-label match
were *not detectably correlated* (r = -0.11, p = 0.63 across 4,705 runs).
This archived result includes the portfolio fixture excluded from corrected
DFAH-Bench v2. Repeatability and correctness require separate evidence.

---

## Quick Start

```bash
pip install -r requirements.txt
python run_dfah_demo.py
```

No LLM or API keys needed. Runs in seconds using deterministic simulation.

> **Note**: The quick demo uses fixed agent logic (no LLM) to demonstrate the DFAH workflow and output format. It measures determinism-style metrics (action, signature, and decision determinism) but does not wire in full faithfulness scoring. For real-LLM evaluation with behavioral variation, use the [benchmark path](#with-a-real-llm) via Ollama.

### Output

Results are saved to `dfah_results/dfah_results.json`:

```json
{
  "dfah_version": "1.0",
  "timestamp": "2026-03-07T...",
  "config": { "n_cases": 5, "n_runs_per_case": 3 },
  "benchmarks": {
    "compliance_triage": {
      "action_determinism": 100.0,
      "signature_determinism": 100.0,
      "decision_determinism": 100.0,
      "accuracy": 80.0,
      "n_tests": 5,
      "n_runs_per_test": 3,
      "ground_truth_distribution": { "dismiss": 3, "escalate": 1, "investigate": 1 }
    }
  }
}
```

### Full Benchmark

```bash
python run_dfah_demo.py --full              # 50 cases, 8 runs (all 3 tasks)
python run_dfah_demo.py --task compliance   # Single task
```

### With a Real LLM

```bash
ollama pull qwen2.5:7b-instruct
python econometrics/benchmarks/run_agentic_benchmark.py \
  --model qwen2.5:7b-instruct --n-cases 5 --n-runs 4
```

---

## Bring Your Own Cases

See [`examples/dfah_custom_task.py`](examples/dfah_custom_task.py) for a working example you can copy and adapt.

The pattern:

```python
from econometrics.agentic.metrics.trajectory_determinism import (
    ToolCall, AgentTrajectory, analyze_trajectory_determinism
)

# 1. Record N trajectories of your agent on the same input
trajectories = []
for i in range(8):
    traj = AgentTrajectory(
        run_id=f"run_{i}",
        input_context={"task_id": "your-task-001"},
        tool_calls=[
            ToolCall(tool_name="your_tool", arguments={"key": "value"}),
        ],
        final_decision="approve",
    )
    trajectories.append(traj)

# 2. Measure determinism
metrics = analyze_trajectory_determinism(trajectories)
print(metrics.summary())
```

### What to Customize

| Component | Where | What to Change |
|-----------|-------|----------------|
| Input cases | `input_context` dict | Your task-specific fields |
| Tool calls | `ToolCall` list | Your agent's actual tool recordings |
| Final decision | `final_decision` field | Your agent's output label/action |
| Pass/fail threshold | Your deployment logic | Gate on `decision_determinism >= 0.90` or similar |
| Benchmark tasks | `econometrics/benchmarks/*/task.py` | Add new task following the existing pattern |

---

## Legacy benchmark fixtures

The earlier runner includes three synthetic fixtures. The corrected DFAH-Bench
analysis retains compliance and DataOps only; portfolio is excluded because its
fixture failed consistency review.

| Task | Decision Space | Tools |
|------|---------------|-------|
| **Compliance Triage** | escalate / dismiss / investigate | check_sanctions, get_customer_profile, calculate_risk_score, search_precedents |
| **Portfolio Constraint** | approve / reject | get_current_holdings, get_market_data, check_position_limit, calculate_sector_exposure |
| **DataOps Exception** | auto_fix / escalate / quarantine | get_exception_details, query_reference_data, get_historical_fixes, validate_fix |

---

## Interpretation

Do not infer hidden reasoning or model type from a replay score. A high
decision-agreement value can coexist with a changing recorded path, and a
stable path can still be wrong. Report the tested task, configuration, replay
count, required channels, and eligibility alongside every measure.

---

## File Reference

| File | Purpose |
|------|---------|
| `run_dfah_demo.py` | Entry point — run this first |
| `dfah_results/dfah_results.json` | Output — structured results |
| `examples/dfah_custom_task.py` | Template — bring your own cases |
| `econometrics/agentic/metrics/trajectory_determinism.py` | Core API — `ToolCall`, `AgentTrajectory`, `analyze_trajectory_determinism` |
| `econometrics/benchmarks/run_agentic_benchmark.py` | LLM runner — test with real models via Ollama |
| `econometrics/benchmarks/*/task.py` | Task definitions — tools, test cases, ground truth |
