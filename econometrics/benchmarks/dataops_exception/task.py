#!/usr/bin/env python3
"""
DataOps Exception Handling Benchmark Task

Evaluates LLM agent determinism and faithfulness on data quality exception
resolution. The agent must decide whether to auto-fix, escalate, or quarantine
data records with quality issues in a financial data pipeline.

Tools:
- get_exception_details(exception_id) - Full exception context
- query_reference_data(field, value) - Look up canonical values
- get_historical_fixes(pattern) - Search past resolutions
- validate_fix(field, old_value, new_value) - Check proposed fix
- apply_fix(exception_id, fix) - Apply and log fix
- escalate_to_human(exception_id, reason) - Escalate to team

Exception Types:
- format_error: Invalid date, number, or identifier format
- reference_mismatch: Ticker, CUSIP, ISIN mismatch
- business_rule: Negative price, impossible date, etc.
- missing_field: Required field is null/empty

Ground Truth Labels:
- auto_fix: Fix can be automatically applied
- escalate: Requires human judgment
- quarantine: Cannot proceed, needs investigation

Metrics:
- Signature Determinism: Same tools with same arguments
- Action Determinism: Same tools called (any order)
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class ExceptionDecision(Enum):
    AUTO_FIX = "auto_fix"
    ESCALATE = "escalate"
    QUARANTINE = "quarantine"


class ExceptionType(Enum):
    FORMAT_ERROR = "format_error"
    REFERENCE_MISMATCH = "reference_mismatch"
    BUSINESS_RULE = "business_rule"
    MISSING_FIELD = "missing_field"


@dataclass
class DataException:
    """A data quality exception in a financial pipeline."""
    exception_id: str
    source: str
    field: str
    value: Any
    exception_type: ExceptionType
    rule_violated: str
    record: Dict[str, Any]
    timestamp: str = ""

    def to_prompt(self) -> str:
        """Format exception for agent prompt."""
        return f"""DATA QUALITY EXCEPTION: {self.exception_id}

Source: {self.source}
Exception Type: {self.exception_type.value}
Field: {self.field}
Invalid Value: {self.value}
Rule Violated: {self.rule_violated}

Full Record:
{json.dumps(self.record, indent=2)}

Please analyze this exception and decide:
1. AUTO_FIX - Apply automatic correction (provide the fix)
2. ESCALATE - Requires human review (provide reason)
3. QUARANTINE - Cannot determine action (needs investigation)

Use the available tools to research the issue and determine the best action."""


@dataclass
class MockDataContext:
    """Simulated data context for deterministic evaluation."""
    reference_data: Dict[str, Dict] = field(default_factory=dict)
    historical_fixes: Dict[str, List[Dict]] = field(default_factory=dict)
    valid_formats: Dict[str, str] = field(default_factory=dict)


class DataOpsTools:
    """Tool implementations for the DataOps exception task."""

    def __init__(self, context: MockDataContext):
        self.context = context
        self.call_log: List[Dict] = []
        self.applied_fixes: List[Dict] = []
        self.escalations: List[Dict] = []

    def get_exception_details(self, exception_id: str) -> Dict:
        """Get full exception details."""
        self.call_log.append({
            "tool": "get_exception_details",
            "args": {"exception_id": exception_id}
        })
        # In real system, would look up exception
        return {
            "exception_id": exception_id,
            "created_at": "2025-01-15T10:30:00Z",
            "source_system": "market_data_feed",
            "priority": "high",
            "sla_deadline": "2025-01-15T11:30:00Z",
            "similar_exceptions_today": 3
        }

    def query_reference_data(self, field: str, value: str) -> Dict:
        """Look up canonical value in reference data."""
        self.call_log.append({
            "tool": "query_reference_data",
            "args": {"field": field, "value": value}
        })
        ref = self.context.reference_data.get(field, {})
        match = ref.get(value.upper() if isinstance(value, str) else str(value))
        return {
            "field": field,
            "query_value": value,
            "canonical_value": match.get("canonical") if match else None,
            "match_found": match is not None,
            "alternatives": match.get("alternatives", []) if match else []
        }

    def get_historical_fixes(self, pattern: str) -> List[Dict]:
        """Search historical fix patterns."""
        self.call_log.append({
            "tool": "get_historical_fixes",
            "args": {"pattern": pattern}
        })
        results = []
        for key, fixes in self.context.historical_fixes.items():
            if pattern.lower() in key.lower():
                results.extend(fixes)
        return results[:5]

    def validate_fix(self, field: str, old_value: Any, new_value: Any) -> Dict:
        """Validate a proposed fix."""
        self.call_log.append({
            "tool": "validate_fix",
            "args": {"field": field, "old_value": old_value, "new_value": new_value}
        })
        # Check format validity
        expected_format = self.context.valid_formats.get(field)
        is_valid = True
        validation_errors = []

        if field == "trade_price" and isinstance(new_value, (int, float)):
            if new_value < 0:
                is_valid = False
                validation_errors.append("Price cannot be negative")
            elif new_value == 0:
                is_valid = False
                validation_errors.append("Price cannot be zero")

        if field == "trade_date":
            try:
                datetime.strptime(str(new_value), "%Y-%m-%d")
            except ValueError:
                is_valid = False
                validation_errors.append("Invalid date format")

        return {
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
            "is_valid": is_valid,
            "validation_errors": validation_errors,
            "expected_format": expected_format
        }

    def apply_fix(self, exception_id: str, fix: Dict) -> Dict:
        """Apply a fix to the exception."""
        self.call_log.append({
            "tool": "apply_fix",
            "args": {"exception_id": exception_id, "fix": fix}
        })
        self.applied_fixes.append({
            "exception_id": exception_id,
            "fix": fix,
            "timestamp": datetime.now().isoformat()
        })
        return {
            "exception_id": exception_id,
            "status": "fixed",
            "fix_applied": fix,
            "audit_log_id": f"AUDIT-{exception_id}"
        }

    def escalate_to_human(self, exception_id: str, reason: str) -> Dict:
        """Escalate exception to human review."""
        self.call_log.append({
            "tool": "escalate_to_human",
            "args": {"exception_id": exception_id, "reason": reason}
        })
        self.escalations.append({
            "exception_id": exception_id,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        return {
            "exception_id": exception_id,
            "status": "escalated",
            "escalation_queue": "data_quality_team",
            "estimated_response": "1 hour"
        }

    def get_tools_schema(self) -> List[Dict]:
        """Return JSON schema for all tools."""
        return [
            {
                "name": "get_exception_details",
                "description": "Get full context about a data exception",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "exception_id": {"type": "string"}
                    },
                    "required": ["exception_id"]
                }
            },
            {
                "name": "query_reference_data",
                "description": "Look up canonical value in reference data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "value": {"type": "string"}
                    },
                    "required": ["field", "value"]
                }
            },
            {
                "name": "get_historical_fixes",
                "description": "Search past fixes for similar exceptions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"}
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "validate_fix",
                "description": "Validate a proposed fix before applying",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "old_value": {},
                        "new_value": {}
                    },
                    "required": ["field", "old_value", "new_value"]
                }
            },
            {
                "name": "apply_fix",
                "description": "Apply and log a fix to the exception",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "exception_id": {"type": "string"},
                        "fix": {"type": "object"}
                    },
                    "required": ["exception_id", "fix"]
                }
            },
            {
                "name": "escalate_to_human",
                "description": "Escalate exception to human review",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "exception_id": {"type": "string"},
                        "reason": {"type": "string"}
                    },
                    "required": ["exception_id", "reason"]
                }
            }
        ]


def create_test_context() -> MockDataContext:
    """Create a mock context for testing."""
    return MockDataContext(
        reference_data={
            "ticker": {
                "MSFT": {"canonical": "MSFT", "alternatives": ["MICROSOFT", "MSFT.O"]},
                "AAPL": {"canonical": "AAPL", "alternatives": ["APPLE", "AAPL.O"]},
                "GOOG": {"canonical": "GOOGL", "alternatives": ["GOOGLE", "GOOG.O"]},
            },
            "cusip": {
                "594918104": {"canonical": "594918104", "name": "MSFT"},
                "037833100": {"canonical": "037833100", "name": "AAPL"},
            }
        },
        historical_fixes={
            "negative_price": [
                {
                    "pattern": "negative_price",
                    "resolution": "absolute_value",
                    "success_rate": 0.95,
                    "count": 150
                }
            ],
            "date_format": [
                {
                    "pattern": "MM/DD/YYYY",
                    "resolution": "convert_to_ISO",
                    "success_rate": 1.0,
                    "count": 500
                }
            ],
            "ticker_mismatch": [
                {
                    "pattern": "alternative_ticker",
                    "resolution": "map_to_canonical",
                    "success_rate": 0.98,
                    "count": 200
                }
            ],
            "missing_cusip": [
                {
                    "pattern": "null_cusip",
                    "resolution": "escalate",
                    "success_rate": 0.0,
                    "count": 50
                }
            ]
        },
        valid_formats={
            "trade_date": "YYYY-MM-DD",
            "trade_price": "positive_decimal",
            "ticker": "1-5_uppercase_letters",
            "cusip": "9_digit_alphanumeric"
        }
    )


# Sample test exceptions
SAMPLE_EXCEPTIONS = [
    DataException(
        exception_id="DQ-2025-00789",
        source="market_data_feed",
        field="trade_price",
        value=-45.23,
        exception_type=ExceptionType.BUSINESS_RULE,
        rule_violated="price_must_be_positive",
        record={
            "ticker": "MSFT",
            "date": "2025-01-15",
            "volume": 12500000,
            "trade_price": -45.23
        }
    ),
    DataException(
        exception_id="DQ-2025-00790",
        source="trade_file",
        field="trade_date",
        value="01/15/2025",
        exception_type=ExceptionType.FORMAT_ERROR,
        rule_violated="date_must_be_ISO_format",
        record={
            "ticker": "AAPL",
            "trade_date": "01/15/2025",
            "quantity": 1000,
            "price": 195.50
        }
    ),
    DataException(
        exception_id="DQ-2025-00791",
        source="corporate_actions",
        field="cusip",
        value=None,
        exception_type=ExceptionType.MISSING_FIELD,
        rule_violated="cusip_required",
        record={
            "ticker": "NEWCO",
            "action_type": "dividend",
            "cusip": None,
            "amount": 0.50
        }
    )
]

# Ground truth for test exceptions
GROUND_TRUTH = {
    "DQ-2025-00789": ExceptionDecision.AUTO_FIX,  # Take absolute value
    "DQ-2025-00790": ExceptionDecision.AUTO_FIX,  # Convert date format
    "DQ-2025-00791": ExceptionDecision.ESCALATE,  # Missing CUSIP needs human
}


def example_dataops_exception():
    """Demonstrate the DataOps exception benchmark."""
    print("=" * 60)
    print("DATAOPS EXCEPTION BENCHMARK - EXAMPLE")
    print("=" * 60)

    context = create_test_context()
    tools = DataOpsTools(context)

    for exc in SAMPLE_EXCEPTIONS:
        print(f"\n{'='*60}")
        print(f"Exception: {exc.exception_id}")
        print(f"Type: {exc.exception_type.value}")
        print(f"Field: {exc.field} = {exc.value}")
        print(f"Rule Violated: {exc.rule_violated}")
        print(f"Ground Truth: {GROUND_TRUTH[exc.exception_id].value}")
        print("-" * 40)

        # Simulate agent tool calls
        tools.call_log = []

        # Get exception details
        details = tools.get_exception_details(exc.exception_id)
        print(f"Exception details: priority={details['priority']}")

        # Search historical fixes
        hist = tools.get_historical_fixes(exc.rule_violated)
        if hist:
            print(f"Historical fixes found: {len(hist)} patterns")
            print(f"  Best match: {hist[0]['resolution']} (success: {hist[0]['success_rate']*100:.0f}%)")

        # For auto-fix cases, validate the fix
        if GROUND_TRUTH[exc.exception_id] == ExceptionDecision.AUTO_FIX:
            if exc.field == "trade_price" and isinstance(exc.value, (int, float)):
                new_value = abs(exc.value)
                validation = tools.validate_fix(exc.field, exc.value, new_value)
                print(f"Proposed fix: {exc.value} -> {new_value}")
                print(f"Validation: {'PASS' if validation['is_valid'] else 'FAIL'}")
            elif exc.field == "trade_date":
                # Convert MM/DD/YYYY to YYYY-MM-DD
                parts = exc.value.split("/")
                new_value = f"{parts[2]}-{parts[0]}-{parts[1]}"
                validation = tools.validate_fix(exc.field, exc.value, new_value)
                print(f"Proposed fix: {exc.value} -> {new_value}")
                print(f"Validation: {'PASS' if validation['is_valid'] else 'FAIL'}")

        # For escalate cases
        if GROUND_TRUTH[exc.exception_id] == ExceptionDecision.ESCALATE:
            ref = tools.query_reference_data(exc.field, str(exc.record.get("ticker", "")))
            print(f"Reference lookup: {'found' if ref['match_found'] else 'not found'}")

        print(f"\nTools called: {[c['tool'] for c in tools.call_log]}")

    print("\n" + "=" * 60)
    print("METRICS TO MEASURE:")
    print("-" * 60)
    print("1. Signature Determinism: Same tools + same arguments")
    print("2. Action Determinism: Same tools called (any order)")
    print("3. Decision Determinism: Same auto_fix/escalate/quarantine")
    print("4. Fix Correctness: Applied fix matches expected")


if __name__ == "__main__":
    example_dataops_exception()
