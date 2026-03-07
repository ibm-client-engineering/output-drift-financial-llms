#!/usr/bin/env python3
"""
Compliance Triage Benchmark Task

Evaluates LLM agent determinism and faithfulness on compliance alert triage.
The agent must decide whether to escalate, dismiss, or investigate alerts
based on evidence from available tools.

Tools:
- search_precedents(query) - Search historical compliance cases
- get_customer_profile(id) - Retrieve customer risk profile
- check_sanctions_list(name) - Screen against sanctions lists
- calculate_risk_score(factors) - Compute transaction risk score

Ground Truth Labels:
- escalate: Alert matches known fraud patterns
- dismiss: Alert is a false positive
- investigate: Insufficient information

Metrics:
- Decision Determinism: Same escalation decision across runs
- Evidence Grounding: Are cited precedents actually retrieved?
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum


class TriageDecision(Enum):
    ESCALATE = "escalate"
    DISMISS = "dismiss"
    INVESTIGATE = "investigate"


@dataclass
class TransactionAlert:
    """A compliance alert for a suspicious transaction."""
    alert_id: str
    amount: float
    currency: str
    sender: str
    receiver: str
    country: str
    flags: List[str]
    description: str = ""

    def to_prompt(self) -> str:
        """Format alert for agent prompt."""
        return f"""COMPLIANCE ALERT: {self.alert_id}

Transaction Details:
- Amount: {self.amount:,.2f} {self.currency}
- Sender: {self.sender}
- Receiver: {self.receiver}
- Destination Country: {self.country}

Flags Triggered: {', '.join(self.flags)}

{f'Additional Context: {self.description}' if self.description else ''}

Please analyze this alert and decide whether to:
1. ESCALATE - Forward to compliance team for investigation
2. DISMISS - Close as false positive
3. INVESTIGATE - Request additional information before deciding

Use the available tools to gather evidence and justify your decision."""


@dataclass
class MockToolContext:
    """Simulated tool responses for deterministic evaluation."""
    sanctions_hits: Dict[str, bool] = field(default_factory=dict)
    customer_profiles: Dict[str, Dict] = field(default_factory=dict)
    precedents: Dict[str, List[Dict]] = field(default_factory=dict)
    risk_scores: Dict[str, float] = field(default_factory=dict)


class ComplianceTriageTools:
    """Tool implementations for the compliance triage task."""

    def __init__(self, context: MockToolContext):
        self.context = context
        self.call_log: List[Dict] = []

    def search_precedents(self, query: str) -> List[Dict]:
        """Search historical compliance cases."""
        self.call_log.append({
            "tool": "search_precedents",
            "args": {"query": query}
        })
        # Return matching precedents based on query keywords
        results = []
        for key, cases in self.context.precedents.items():
            if any(kw.lower() in query.lower() for kw in key.split()):
                results.extend(cases)
        return results[:5]  # Limit to 5 results

    def get_customer_profile(self, customer_id: str) -> Dict:
        """Retrieve customer risk profile."""
        self.call_log.append({
            "tool": "get_customer_profile",
            "args": {"customer_id": customer_id}
        })
        return self.context.customer_profiles.get(customer_id, {
            "id": customer_id,
            "risk_level": "unknown",
            "kyc_status": "incomplete",
            "relationship_years": 0
        })

    def check_sanctions_list(self, name: str) -> Dict:
        """Screen against sanctions lists."""
        self.call_log.append({
            "tool": "check_sanctions_list",
            "args": {"name": name}
        })
        is_hit = self.context.sanctions_hits.get(name.lower(), False)
        return {
            "name": name,
            "is_sanctioned": is_hit,
            "list_type": "OFAC" if is_hit else None,
            "match_score": 1.0 if is_hit else 0.0
        }

    def calculate_risk_score(self, factors: Dict) -> Dict:
        """Compute transaction risk score."""
        self.call_log.append({
            "tool": "calculate_risk_score",
            "args": {"factors": factors}
        })
        # Deterministic score based on factors
        score = 0.0
        if factors.get("amount", 0) > 50000:
            score += 0.3
        if factors.get("offshore", False):
            score += 0.25
        if factors.get("new_counterparty", False):
            score += 0.2
        if factors.get("sanctions_hit", False):
            score += 0.4
        return {
            "risk_score": min(score, 1.0),
            "risk_level": "high" if score > 0.6 else "medium" if score > 0.3 else "low",
            "factors_considered": list(factors.keys())
        }

    def get_tools_schema(self) -> List[Dict]:
        """Return JSON schema for all tools."""
        return [
            {
                "name": "search_precedents",
                "description": "Search historical compliance cases for similar alerts",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for precedent cases"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_customer_profile",
                "description": "Retrieve customer risk profile and KYC status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "Customer identifier"
                        }
                    },
                    "required": ["customer_id"]
                }
            },
            {
                "name": "check_sanctions_list",
                "description": "Screen entity name against sanctions lists",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Entity name to screen"
                        }
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "calculate_risk_score",
                "description": "Compute risk score based on transaction factors",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "factors": {
                            "type": "object",
                            "description": "Risk factors dictionary"
                        }
                    },
                    "required": ["factors"]
                }
            }
        ]


def create_test_context() -> MockToolContext:
    """Create a mock context for testing."""
    return MockToolContext(
        sanctions_hits={
            "xyz holdings": False,
            "shadow corp": True,
            "legitimate inc": False
        },
        customer_profiles={
            "ABC Corp": {
                "id": "ABC Corp",
                "risk_level": "medium",
                "kyc_status": "complete",
                "relationship_years": 5,
                "transaction_history": "regular wire transfers"
            },
            "New Customer LLC": {
                "id": "New Customer LLC",
                "risk_level": "high",
                "kyc_status": "pending",
                "relationship_years": 0,
                "transaction_history": "no history"
            }
        },
        precedents={
            "offshore wire unusual": [
                {
                    "case_id": "CASE-2024-0892",
                    "outcome": "dismissed",
                    "reason": "Regular business payment to overseas supplier",
                    "similarity": 0.85
                }
            ],
            "sanctions hit": [
                {
                    "case_id": "CASE-2023-1234",
                    "outcome": "escalated",
                    "reason": "Direct sanctions list match",
                    "similarity": 0.95
                }
            ],
            "new customer large": [
                {
                    "case_id": "CASE-2024-0567",
                    "outcome": "investigated",
                    "reason": "Insufficient KYC for transaction size",
                    "similarity": 0.78
                }
            ]
        }
    )


# Sample test alerts
SAMPLE_ALERTS = [
    TransactionAlert(
        alert_id="TXN-2025-001234",
        amount=47500.00,
        currency="USD",
        sender="ABC Corp",
        receiver="XYZ Holdings",
        country="Cayman Islands",
        flags=["unusual_amount", "offshore_destination"],
        description="Wire transfer flagged for offshore destination"
    ),
    TransactionAlert(
        alert_id="TXN-2025-001235",
        amount=125000.00,
        currency="USD",
        sender="New Customer LLC",
        receiver="Shadow Corp",
        country="Belarus",
        flags=["new_customer", "high_risk_country", "large_amount"],
        description="First transaction from new customer to high-risk jurisdiction"
    ),
    TransactionAlert(
        alert_id="TXN-2025-001236",
        amount=5000.00,
        currency="USD",
        sender="Legitimate Inc",
        receiver="Supplier Co",
        country="Canada",
        flags=["round_amount"],
        description="Round amount wire transfer"
    )
]

# Ground truth for test alerts
GROUND_TRUTH = {
    "TXN-2025-001234": TriageDecision.DISMISS,  # Regular offshore payment
    "TXN-2025-001235": TriageDecision.ESCALATE,  # Sanctions + new customer + high risk
    "TXN-2025-001236": TriageDecision.DISMISS,   # Minor flag, low amount
}


def example_compliance_triage():
    """Demonstrate the compliance triage benchmark."""
    print("=" * 60)
    print("COMPLIANCE TRIAGE BENCHMARK - EXAMPLE")
    print("=" * 60)

    context = create_test_context()
    tools = ComplianceTriageTools(context)

    for alert in SAMPLE_ALERTS:
        print(f"\n{'='*60}")
        print(f"Alert: {alert.alert_id}")
        print(f"Amount: {alert.amount:,.2f} {alert.currency}")
        print(f"Sender: {alert.sender} -> Receiver: {alert.receiver}")
        print(f"Flags: {alert.flags}")
        print(f"Ground Truth: {GROUND_TRUTH[alert.alert_id].value}")
        print("-" * 40)

        # Simulate agent tool calls
        tools.call_log = []

        # Example tool sequence
        sanctions = tools.check_sanctions_list(alert.receiver)
        print(f"Sanctions check: {sanctions['name']} -> {sanctions['is_sanctioned']}")

        profile = tools.get_customer_profile(alert.sender)
        print(f"Customer profile: {profile['risk_level']} risk, {profile['kyc_status']} KYC")

        precedents = tools.search_precedents(" ".join(alert.flags))
        print(f"Precedents found: {len(precedents)}")

        risk = tools.calculate_risk_score({
            "amount": alert.amount,
            "offshore": alert.country not in ["USA", "Canada", "UK"],
            "new_counterparty": profile.get("relationship_years", 0) == 0,
            "sanctions_hit": sanctions["is_sanctioned"]
        })
        print(f"Risk score: {risk['risk_score']:.2f} ({risk['risk_level']})")

        print(f"\nTools called: {[c['tool'] for c in tools.call_log]}")

    print("\n" + "=" * 60)
    print("METRICS TO MEASURE:")
    print("-" * 60)
    print("1. Decision Determinism: Same decision across N runs")
    print("2. Action Determinism: Same tools called (any order)")
    print("3. Signature Determinism: Same tools with same args")
    print("4. Evidence Grounding: Citations match retrieved data")


if __name__ == "__main__":
    example_compliance_triage()
