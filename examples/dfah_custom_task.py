#!/usr/bin/env python3
"""
DFAH: Bring Your Own Cases

Shows how to use the DFAH trajectory determinism metrics with your own
agent trajectories. Replace the example data below with recordings from
your own LLM agent system.

Usage:
    python examples/dfah_custom_task.py

What this measures:
    - Action Determinism:    Did the agent call the same tools each run?
    - Signature Determinism: Did it call them with the same arguments?
    - Decision Determinism:  Did it reach the same final decision?

See also: DFAH.md for full documentation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from econometrics.agentic.metrics.trajectory_determinism import (
    ToolCall,
    AgentTrajectory,
    analyze_trajectory_determinism,
)


def main():
    # ------------------------------------------------------------------
    # Step 1: Record trajectories from your agent
    #
    # Each AgentTrajectory captures one run of your agent on one input.
    # Run the same input N times and collect the results.
    # ------------------------------------------------------------------

    input_context = {
        "alert_id": "TXN-001",
        "amount": 75000,
        "sender": "Acme Corp",
        "receiver": "Offshore LLC",
        "country": "Cayman Islands",
    }

    # Simulate 8 runs. In practice, replace this with actual agent outputs.
    trajectories = []
    for i in range(8):
        traj = AgentTrajectory(
            run_id=f"run_{i}",
            input_context=input_context,
            tool_calls=[
                ToolCall(
                    tool_name="check_sanctions",
                    arguments={"entity": "Offshore LLC"},
                ),
                ToolCall(
                    tool_name="get_customer_profile",
                    arguments={"customer_id": "Acme Corp"},
                ),
                ToolCall(
                    tool_name="calculate_risk_score",
                    arguments={"amount": 75000, "offshore": True},
                ),
            ],
            final_decision="escalate",
        )
        trajectories.append(traj)

    # ------------------------------------------------------------------
    # Step 2: Analyze determinism
    # ------------------------------------------------------------------

    metrics = analyze_trajectory_determinism(trajectories)
    print(metrics.summary())

    # ------------------------------------------------------------------
    # Step 3: Use individual scores in your own logic
    # ------------------------------------------------------------------

    print("Quick access to scores:")
    print(f"  Action Determinism:    {metrics.action_determinism:.1%}")
    print(f"  Signature Determinism: {metrics.signature_determinism:.1%}")
    print(f"  Decision Determinism:  {metrics.decision_determinism:.1%}")
    print()

    # Example: gate deployment on determinism threshold
    THRESHOLD = 0.90
    if metrics.decision_determinism >= THRESHOLD:
        print(f"  PASS: decision determinism >= {THRESHOLD:.0%}")
    else:
        print(f"  FAIL: decision determinism < {THRESHOLD:.0%} — review before deploying")
    print()

    # ------------------------------------------------------------------
    # Customization checklist:
    #   1. Replace input_context with your task input
    #   2. Replace tool_calls with your agent's actual tool call recordings
    #   3. Replace final_decision with your agent's output
    #   4. Run your agent N times (8 is recommended) per input
    #   5. Set your own THRESHOLD for pass/fail
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
