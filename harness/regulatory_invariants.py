#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Historical control-invariant examples for the financial AI workshop.

This legacy module records illustrative benchmark checks and possible regulatory
touchpoints. It is not legal guidance, does not implement the cited rules, and
does not determine compliance. Numeric defaults must be replaced with values
approved for the intended workflow.

The source names below are retained as possible governance touchpoints from the
original workshop. They do not establish that a rule applies to a given system
or that the numerical profiles implement that rule.

AI4F Workshop 2025: "LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows"
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import math


# =============================================================================
# HISTORICAL WORKSHOP DEFAULTS
# =============================================================================

# The workshop used 5% as an illustrative numeric comparison tolerance. There is
# no universal 5% GAAP materiality threshold.
DEFAULT_NUMERIC_TOLERANCE: float = 0.05
# Deprecated compatibility alias. The name must not be read as legal guidance.
GAAP_MATERIALITY_THRESHOLD: float = DEFAULT_NUMERIC_TOLERANCE

# Illustrative 1% numeric profile retained for backward compatibility.
BASEL_III_RISK_TOLERANCE: float = 0.01

# Exact-output workshop target retained for backward compatibility.
FSB_IDENTITY_REQUIREMENT: float = 1.0

# Illustrative source-reference match target retained for compatibility.
SEC_CITATION_ACCURACY_THRESHOLD: float = 0.95

# Historical workshop trace-completeness target.
CFTC_AUDIT_COMPLETENESS_THRESHOLD: float = 1.0


# =============================================================================
# POSSIBLE GOVERNANCE TOUCHPOINT LABELS
# =============================================================================

class RegulatoryBody(Enum):
    """Legacy source labels used by the workshop profiles."""
    FSB = "Financial Stability Board"
    BIS = "Bank for International Settlements"
    CFTC = "Commodity Futures Trading Commission"
    SEC = "Securities and Exchange Commission"
    GAAP = "Generally Accepted Accounting Principles"
    EU_AI_ACT = "European Union AI Act"
    FINRA = "Financial Industry Regulatory Authority"
    OCC = "Office of the Comptroller of the Currency"
    BENCHMARK = "Illustrative benchmark configuration"


@dataclass(frozen=True)
class RegulatoryRequirement:
    """
    Immutable historical workshop control profile.

    Attributes:
        body: Possible governance touchpoint or benchmark label
        rule_id: Legacy profile identifier
        requirement_name: Human-readable name
        description: Detailed description of the workshop profile
        threshold: Configured benchmark threshold (if applicable)
        threshold_type: How to interpret the threshold ('min', 'max', 'exact')
        citation: Potential source touchpoint or benchmark note
    """
    body: RegulatoryBody
    rule_id: str
    requirement_name: str
    description: str
    threshold: Optional[float]
    threshold_type: str  # 'min', 'max', 'exact', 'range'
    citation: str


# =============================================================================
# HISTORICAL WORKSHOP CONTROL PROFILES
# =============================================================================

REGULATORY_REQUIREMENTS: Dict[str, RegulatoryRequirement] = {
    # Legacy keys are preserved for wire compatibility. The descriptions define
    # benchmark profiles, not regulator-issued requirements.
    "fsb_consistent_decisions": RegulatoryRequirement(
        body=RegulatoryBody.BENCHMARK,
        rule_id="WORKSHOP-EXACT-OUTPUT",
        requirement_name="Exact-Output Replay Profile",
        description=(
            "The workshop profile checks exact output identity across repeated "
            "inputs. BCBS 239 does not prescribe identical LLM outputs."
        ),
        threshold=FSB_IDENTITY_REQUIREMENT,
        threshold_type="min",
        citation="Historical workshop profile; not a BCBS 239 requirement"
    ),

    "fsb_adaptability": RegulatoryRequirement(
        body=RegulatoryBody.BENCHMARK,
        rule_id="WORKSHOP-ADAPTABILITY",
        requirement_name="Adaptability Documentation Profile",
        description=(
            "The workshop records whether a configuration can be changed while "
            "preserving the declared replay contract."
        ),
        threshold=None,
        threshold_type="qualitative",
        citation="Historical workshop profile; not a BCBS 239 requirement"
    ),

    "bis_risk_calculation_tolerance": RegulatoryRequirement(
        body=RegulatoryBody.BENCHMARK,
        rule_id="WORKSHOP-NUMERIC-REPLAY",
        requirement_name="Illustrative Numeric-Replay Profile",
        description=(
            "The workshop applies a configurable 1% comparison band to selected "
            "numeric examples. It is not a Basel-prescribed universal tolerance."
        ),
        threshold=BASEL_III_RISK_TOLERANCE,
        threshold_type="max",
        citation="Historical workshop tolerance; not a Basel requirement"
    ),

    "cftc_audit_trail": RegulatoryRequirement(
        body=RegulatoryBody.BENCHMARK,
        rule_id="WORKSHOP-TRACE-FIELDS",
        requirement_name="Trace-Field Completeness Profile",
        description=(
            "The workshop checks whether a configured list of trace fields is "
            "present. It does not determine CFTC recordkeeping compliance."
        ),
        threshold=CFTC_AUDIT_COMPLETENESS_THRESHOLD,
        threshold_type="min",
        citation="Historical workshop trace-field profile; not a recordkeeping determination"
    ),

    "cftc_ai_documentation": RegulatoryRequirement(
        body=RegulatoryBody.BENCHMARK,
        rule_id="WORKSHOP-MODEL-DOCUMENTATION",
        requirement_name="Model Documentation Profile",
        description=(
            "The workshop records selected model, validation, and monitoring "
            "metadata for later review."
        ),
        threshold=None,
        threshold_type="qualitative",
        citation="Historical workshop model-documentation profile"
    ),

    "sec_citation_accuracy": RegulatoryRequirement(
        body=RegulatoryBody.BENCHMARK,
        rule_id="WORKSHOP-SOURCE-MATCH",
        requirement_name="Source-Reference Match Profile",
        description=(
            "The workshop checks whether extracted citation identifiers occur in "
            "the supplied source list. It does not determine a Rule 10b-5 violation."
        ),
        threshold=SEC_CITATION_ACCURACY_THRESHOLD,
        threshold_type="min",
        citation="Historical workshop source-reference profile; not a legal determination"
    ),

    "sec_record_retention": RegulatoryRequirement(
        body=RegulatoryBody.BENCHMARK,
        rule_id="WORKSHOP-RECORD-CAPTURE",
        requirement_name="Record-Capture Profile",
        description=(
            "The workshop records selected replay metadata. It does not determine "
            "record scope, format, retention period, or Rule 17a-4 compliance."
        ),
        threshold=None,
        threshold_type="qualitative",
        citation="Historical workshop record-capture profile; not a retention determination"
    ),

    # Legacy key retained for compatibility. This is an exercise setting, not a
    # GAAP requirement or materiality determination.
    "gaap_materiality": RegulatoryRequirement(
        body=RegulatoryBody.BENCHMARK,
        rule_id="EXERCISE-NUMERIC-TOLERANCE",
        requirement_name="Illustrative Numeric Tolerance",
        description=(
            "The workshop compares selected numeric outputs using a configurable "
            "5% default. This value is not a universal accounting threshold."
        ),
        threshold=GAAP_MATERIALITY_THRESHOLD,
        threshold_type="max",
        citation="Illustrative benchmark setting; configure per approved task contract"
    ),

    "eu_ai_act_high_risk": RegulatoryRequirement(
        body=RegulatoryBody.BENCHMARK,
        rule_id="WORKSHOP-CLASSIFICATION-REVIEW",
        requirement_name="High-Risk Classification Review Profile",
        description=(
            "The workshop flags a task for independent scope and classification "
            "review; the task label alone does not determine legal status."
        ),
        threshold=None,
        threshold_type="qualitative",
        citation="Historical workshop classification-review profile; obtain independent legal scope review"
    ),

    "eu_ai_act_transparency": RegulatoryRequirement(
        body=RegulatoryBody.BENCHMARK,
        rule_id="WORKSHOP-TRANSPARENCY-METADATA",
        requirement_name="Transparency Metadata Profile",
        description=(
            "The workshop checks whether selected traceability metadata is "
            "captured for later review."
        ),
        threshold=None,
        threshold_type="qualitative",
        citation="Historical workshop transparency-metadata profile"
    ),
}


# =============================================================================
# TASK-SPECIFIC WORKSHOP PROFILE MAPPINGS
# =============================================================================

# Legacy mapping of each task type to profiles the workshop evaluates. This does
# not assert that every named source applies to the task.
TASK_REGULATORY_MAPPINGS: Dict[str, List[str]] = {
    "rag": [
        "fsb_consistent_decisions",      # Exact-output workshop profile
        "sec_citation_accuracy",         # Source-reference match profile
        "sec_record_retention",          # Record-capture profile
        "eu_ai_act_transparency",        # Transparency-metadata profile
    ],
    "sql": [
        "gaap_materiality",              # Legacy key: configurable numeric tolerance
        "bis_risk_calculation_tolerance", # Numeric-replay profile
        "cftc_audit_trail",              # Trace-field profile
        "fsb_consistent_decisions",      # Exact-output profile
    ],
    "summary": [
        "fsb_consistent_decisions",      # Exact-output profile
        "sec_record_retention",          # Record-capture profile
        "eu_ai_act_high_risk",           # Classification-review profile
        "cftc_ai_documentation",         # Model-documentation profile
    ],
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_gaap_materiality(
    actual_value: float,
    expected_value: float,
    threshold: float = GAAP_MATERIALITY_THRESHOLD
) -> Dict[str, Any]:
    """
    Compare a numeric output with the configured exercise tolerance.

    The function name is retained for backward compatibility. Its result does
    not determine GAAP materiality or regulatory compliance.

    Args:
        actual_value: The value produced by the AI system
        expected_value: The expected/correct value
        threshold: Task-specific tolerance (default: 5% for the exercise)

    Returns:
        Dict with the tolerance result and compatibility metadata
    """
    if expected_value == 0:
        deviation_pct = float('inf') if actual_value != 0 else 0.0
    else:
        deviation_pct = abs(actual_value - expected_value) / abs(expected_value)

    is_compliant = deviation_pct <= threshold

    return {
        "compliant": is_compliant,
        "actual_value": actual_value,
        "expected_value": expected_value,
        "deviation_pct": deviation_pct,
        "threshold_pct": threshold,
        "regulatory_basis": REGULATORY_REQUIREMENTS["gaap_materiality"],
        "requirement_id": "gaap_materiality",
        "within_tolerance": is_compliant,
        "regulatory_body": RegulatoryBody.BENCHMARK.value,
        "rule_citation": "Illustrative task tolerance; configure per approved workflow",
        "interpretation": "Illustrative workshop profile; not legal or regulatory compliance",
    }


def validate_fsb_consistency(
    outputs: List[Any],
    require_identity: bool = True
) -> Dict[str, Any]:
    """
    Apply the workshop's exact-output replay profile.

    The function and field names are retained for compatibility. Passing this
    check does not establish compliance with BCBS 239 or any other rule.

    Args:
        outputs: List of outputs from identical inputs. Every observation must
            be a non-blank string.
        require_identity: If True, requires 100% identity; otherwise reports rate

    Returns:
        Dict with profile status and consistency metrics
    """
    invalid_output_indices = [
        index
        for index, output in enumerate(outputs)
        if not isinstance(output, str) or not output.strip()
    ]
    if not outputs or invalid_output_indices:
        status = "not_evaluated" if not outputs else "invalid_input"
        error = (
            "No outputs provided for consistency validation"
            if not outputs
            else "Outputs must be non-blank strings for consistency validation"
        )
        return {
            "compliant": False,
            "passed_profile": False,
            "status": status,
            "error": error,
            "identity_rate": None,
            "total_outputs": len(outputs),
            "identical_outputs": 0,
            "valid_output_count": len(outputs) - len(invalid_output_indices),
            "invalid_output_count": len(invalid_output_indices),
            "invalid_output_indices": invalid_output_indices,
            "threshold": FSB_IDENTITY_REQUIREMENT,
            "regulatory_basis": REGULATORY_REQUIREMENTS["fsb_consistent_decisions"],
            "requirement_id": "fsb_consistent_decisions",
            "regulatory_body": RegulatoryBody.BENCHMARK.value,
            "rule_citation": "Historical workshop exact-output profile",
            "interpretation": "Illustrative workshop profile; not legal or regulatory compliance",
        }

    reference = outputs[0]
    identical_count = sum(1 for o in outputs if o == reference)
    identity_rate = identical_count / len(outputs)

    is_compliant = identity_rate >= FSB_IDENTITY_REQUIREMENT if require_identity else True

    return {
        "compliant": is_compliant,
        "status": "passed" if is_compliant else "failed",
        "identity_rate": identity_rate,
        "total_outputs": len(outputs),
        "identical_outputs": identical_count,
        "valid_output_count": len(outputs),
        "invalid_output_count": 0,
        "invalid_output_indices": [],
        "threshold": FSB_IDENTITY_REQUIREMENT,
        "regulatory_basis": REGULATORY_REQUIREMENTS["fsb_consistent_decisions"],
        "requirement_id": "fsb_consistent_decisions",
        "passed_profile": is_compliant,
        "regulatory_body": RegulatoryBody.BENCHMARK.value,
        "rule_citation": "Historical workshop exact-output profile",
        "interpretation": "Illustrative workshop profile; not legal or regulatory compliance",
    }


def validate_sec_citations(
    citations: List[str],
    available_sources: List[str],
    threshold: float = SEC_CITATION_ACCURACY_THRESHOLD
) -> Dict[str, Any]:
    """
    Apply the workshop's source-reference match profile.

    This check compares identifiers against supplied sources. It does not
    establish factual accuracy or determine a Rule 10b-5 violation.

    Args:
        citations: List of citations extracted from AI output
        available_sources: List of valid source document identifiers
        threshold: Configured source-match threshold (default: 95% in the workshop)

    Returns:
        Dict with profile status and citation analysis
    """
    if not citations:
        # With no citations, there are no identifiers to compare. This says
        # nothing about factual accuracy or legal compliance.
        return {
            "compliant": True,
            "citation_accuracy": 1.0,
            "total_citations": 0,
            "valid_citations": [],
            "invalid_citations": [],
            "regulatory_basis": REGULATORY_REQUIREMENTS["sec_citation_accuracy"],
            "requirement_id": "sec_citation_accuracy",
            "passed_profile": True,
            "regulatory_body": RegulatoryBody.BENCHMARK.value,
            "rule_citation": "Historical workshop source-reference profile",
            "interpretation": "Illustrative workshop profile; not factual, legal, or regulatory compliance",
        }

    # Normalize source names (handle with/without .txt extension)
    normalized_sources = set()
    for source in available_sources:
        normalized_sources.add(source)
        if source.endswith('.txt'):
            normalized_sources.add(source[:-4])
        else:
            normalized_sources.add(source + '.txt')

    valid_citations = [c for c in citations if c in normalized_sources]
    invalid_citations = [c for c in citations if c not in normalized_sources]
    accuracy = len(valid_citations) / len(citations)

    is_compliant = accuracy >= threshold

    return {
        "compliant": is_compliant,
        "citation_accuracy": accuracy,
        "total_citations": len(citations),
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "threshold": threshold,
        "regulatory_basis": REGULATORY_REQUIREMENTS["sec_citation_accuracy"],
        "requirement_id": "sec_citation_accuracy",
        "passed_profile": is_compliant,
        "regulatory_body": RegulatoryBody.BENCHMARK.value,
        "rule_citation": "Historical workshop source-reference profile",
        "interpretation": "Illustrative workshop profile; not factual, legal, or regulatory compliance",
    }


def validate_cftc_audit_trail(
    trace_record: Dict[str, Any],
    required_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Apply the workshop's configured trace-field completeness profile.

    This presence check does not test immutability, retention, record scope, or
    compliance with CFTC 17 CFR 1.31.

    Args:
        trace_record: The audit trail record to validate
        required_fields: List of required field names (defaults to standard set)

    Returns:
        Dict with profile status and field analysis
    """
    if required_fields is None:
        required_fields = [
            "timestamp",      # When the decision was made
            "model",          # Which model made the decision
            "prompt",         # Input to the model
            "output",         # Model's output
            "temperature",    # Sampling parameters
            "seed",           # Reproducibility seed
        ]

    present_fields = [f for f in required_fields if f in trace_record]
    missing_fields = [f for f in required_fields if f not in trace_record]
    completeness = len(present_fields) / len(required_fields)

    is_compliant = completeness >= CFTC_AUDIT_COMPLETENESS_THRESHOLD

    return {
        "compliant": is_compliant,
        "completeness": completeness,
        "present_fields": present_fields,
        "missing_fields": missing_fields,
        "threshold": CFTC_AUDIT_COMPLETENESS_THRESHOLD,
        "regulatory_basis": REGULATORY_REQUIREMENTS["cftc_audit_trail"],
        "requirement_id": "cftc_audit_trail",
        "passed_profile": is_compliant,
        "regulatory_body": RegulatoryBody.BENCHMARK.value,
        "rule_citation": "Historical workshop trace-field profile",
        "interpretation": "Illustrative workshop profile; not legal or regulatory compliance",
    }


# =============================================================================
# COMPOSITE VALIDATION
# =============================================================================

def validate_task_compliance(
    task_type: str,
    validation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Aggregate legacy-named workshop profile results for a task.

    The function name and ``compliant`` fields are compatibility aliases. The
    aggregate is not a legal or regulatory compliance determination.

    Args:
        task_type: One of 'rag', 'sql', 'summary'
        validation_results: Dict mapping requirement_id to validation result

    Returns:
        Composite profile report with legacy compatibility fields
    """
    if task_type not in TASK_REGULATORY_MAPPINGS:
        return {
            "task_type": task_type,
            "overall_compliant": False,
            "all_profiles_passed": False,
            "status": "invalid_task_type",
            "error": f"Unknown task type: {task_type!r}",
            "supported_task_types": sorted(TASK_REGULATORY_MAPPINGS),
            "applicable_requirements": [],
            "validation_results": {},
            "not_evaluated_requirements": [],
            "interpretation": "Illustrative workshop profiles; not legal or regulatory compliance",
            "regulatory_bodies_involved": [],
        }

    applicable_requirements = TASK_REGULATORY_MAPPINGS[task_type]

    compliance_results = {}
    all_compliant = True
    not_evaluated_requirements = []

    for req_id in applicable_requirements:
        if req_id in validation_results:
            result = validation_results[req_id]
            compliance_results[req_id] = result
            is_not_evaluated = (
                result.get("compliant") is None
                or result.get("status") == "not_evaluated"
            )
            if is_not_evaluated:
                all_compliant = False
                not_evaluated_requirements.append(req_id)
            elif result.get("compliant") is not True:
                all_compliant = False
        else:
            compliance_results[req_id] = {
                "compliant": None,
                "status": "not_evaluated",
                "requirement": REGULATORY_REQUIREMENTS.get(req_id)
            }
            not_evaluated_requirements.append(req_id)
            all_compliant = False

    if not_evaluated_requirements:
        status = "incomplete"
    elif all_compliant:
        status = "passed"
    else:
        status = "failed"

    return {
        "task_type": task_type,
        "overall_compliant": all_compliant,
        "all_profiles_passed": all_compliant,
        "status": status,
        "applicable_requirements": applicable_requirements,
        "validation_results": compliance_results,
        "not_evaluated_requirements": not_evaluated_requirements,
        "interpretation": "Illustrative workshop profiles; not legal or regulatory compliance",
        "regulatory_bodies_involved": list(set(
            REGULATORY_REQUIREMENTS[req_id].body.value
            for req_id in applicable_requirements
            if req_id in REGULATORY_REQUIREMENTS
        ))
    }


def get_regulatory_metadata_for_task(task_type: str) -> Dict[str, Any]:
    """
    Get legacy workshop-profile metadata for trace entries.

    The metadata names possible governance touchpoints. It does not establish
    applicability or compliance.

    Args:
        task_type: One of 'rag', 'sql', 'summary'

    Returns:
        Dict with regulatory metadata for the task
    """
    applicable_req_ids = TASK_REGULATORY_MAPPINGS.get(task_type, [])

    requirements_detail = []
    for req_id in applicable_req_ids:
        if req_id in REGULATORY_REQUIREMENTS:
            req = REGULATORY_REQUIREMENTS[req_id]
            requirements_detail.append({
                "requirement_id": req_id,
                "regulatory_body": req.body.value,
                "rule_id": req.rule_id,
                "requirement_name": req.requirement_name,
                "threshold": req.threshold,
                "citation": req.citation
            })

    return {
        "regulatory_framework": {
            "task_type": task_type,
            "applicable_requirements": applicable_req_ids,
            "requirements_detail": requirements_detail,
            "compliance_standard": "AI4F_2025_Financial_AI",
            "interpretation": (
                "Historical workshop profile metadata; not legal or "
                "regulatory compliance"
            ),
        }
    }
