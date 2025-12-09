#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regulatory Invariants for Financial AI Compliance.

This module encodes finance-specific validation logic with explicit mappings to
regulatory requirements from FSB, BIS, CFTC, and other financial oversight bodies.

The invariants defined here are NOT generic ML reproducibility measures—they are
calibrated to specific regulatory thresholds and compliance requirements for
financial services AI deployments.

Regulatory Framework References:
    - FSB (Financial Stability Board): "Principles for Effective Risk Data Aggregation"
      requires "consistent decisions" across identical inputs for regulatory reporting.
    - BIS (Bank for International Settlements): Basel III framework requires
      reproducible risk calculations within defined tolerance bands.
    - CFTC (Commodity Futures Trading Commission): Rule 17a-4 mandates complete
      audit trails for automated trading decisions.
    - SEC Rule 17a-4: Requires broker-dealers to preserve records in non-rewritable format.
    - GAAP ASC 450: Materiality threshold of 5% for financial statement disclosures.
    - EU AI Act (2024): High-risk AI systems require documented decision rationale.

ACM ICAIF 2025: "LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows"
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import math


# =============================================================================
# REGULATORY THRESHOLD CONSTANTS
# =============================================================================

# GAAP ASC 450-20 Materiality Threshold
# SEC Staff Accounting Bulletin No. 99 establishes 5% as the de facto materiality
# threshold for financial statement disclosures. Deviations below this threshold
# are considered immaterial and do not require disclosure or restatement.
GAAP_MATERIALITY_THRESHOLD: float = 0.05  # 5% tolerance for numerical outputs

# Basel III Operational Risk Tolerance
# BIS BCBS 239 (Principles for effective risk data aggregation) requires
# consistency in risk calculations. This tolerance applies to aggregated metrics.
BASEL_III_RISK_TOLERANCE: float = 0.01  # 1% tolerance for risk calculations

# FSB Consistency Requirement
# FSB "Principles for Sound Practices in Operational Resilience" requires
# that identical inputs produce identical outputs for regulatory reporting.
FSB_IDENTITY_REQUIREMENT: float = 1.0  # 100% identity rate required

# SEC Citation Accuracy Threshold
# SEC Rule 10b-5 anti-fraud provisions require accurate source attribution.
# Citations must map to actual provided documents with high accuracy.
SEC_CITATION_ACCURACY_THRESHOLD: float = 0.95  # 95% citation validity required

# CFTC Audit Trail Completeness
# CFTC Rule 17a-4 requires complete, immutable audit trails for all
# automated trading and advisory decisions.
CFTC_AUDIT_COMPLETENESS_THRESHOLD: float = 1.0  # 100% trace coverage required


# =============================================================================
# REGULATORY REQUIREMENT MAPPINGS
# =============================================================================

class RegulatoryBody(Enum):
    """Financial regulatory bodies with jurisdiction over AI systems."""
    FSB = "Financial Stability Board"
    BIS = "Bank for International Settlements"
    CFTC = "Commodity Futures Trading Commission"
    SEC = "Securities and Exchange Commission"
    GAAP = "Generally Accepted Accounting Principles"
    EU_AI_ACT = "European Union AI Act"
    FINRA = "Financial Industry Regulatory Authority"
    OCC = "Office of the Comptroller of the Currency"


@dataclass(frozen=True)
class RegulatoryRequirement:
    """
    Immutable specification of a regulatory compliance requirement.

    Attributes:
        body: Regulatory authority issuing the requirement
        rule_id: Specific rule or principle identifier
        requirement_name: Human-readable name
        description: Detailed description of the requirement
        threshold: Numeric threshold for compliance (if applicable)
        threshold_type: How to interpret the threshold ('min', 'max', 'exact')
        citation: Official document citation
    """
    body: RegulatoryBody
    rule_id: str
    requirement_name: str
    description: str
    threshold: Optional[float]
    threshold_type: str  # 'min', 'max', 'exact', 'range'
    citation: str


# =============================================================================
# FINANCIAL AI COMPLIANCE REQUIREMENTS DATABASE
# =============================================================================

REGULATORY_REQUIREMENTS: Dict[str, RegulatoryRequirement] = {
    # FSB Requirements
    "fsb_consistent_decisions": RegulatoryRequirement(
        body=RegulatoryBody.FSB,
        rule_id="BCBS-239-P6",
        requirement_name="Consistent Decision Outputs",
        description=(
            "Identical inputs must produce identical outputs for regulatory reporting. "
            "This ensures that risk aggregation and regulatory submissions are reproducible "
            "across time and systems."
        ),
        threshold=FSB_IDENTITY_REQUIREMENT,
        threshold_type="min",
        citation="FSB Principles for Effective Risk Data Aggregation (2013), Principle 6: Accuracy"
    ),

    "fsb_adaptability": RegulatoryRequirement(
        body=RegulatoryBody.FSB,
        rule_id="BCBS-239-P11",
        requirement_name="Adaptability and Flexibility",
        description=(
            "Data aggregation systems must be adaptable to changes in business needs, "
            "regulations, and risk management practices while maintaining consistency."
        ),
        threshold=None,
        threshold_type="qualitative",
        citation="FSB Principles for Effective Risk Data Aggregation (2013), Principle 11"
    ),

    # BIS/Basel Requirements
    "bis_risk_calculation_tolerance": RegulatoryRequirement(
        body=RegulatoryBody.BIS,
        rule_id="BCBS-457",
        requirement_name="Risk Calculation Reproducibility",
        description=(
            "Risk-weighted asset calculations must be reproducible within defined tolerance "
            "bands to ensure comparability across institutions and time periods."
        ),
        threshold=BASEL_III_RISK_TOLERANCE,
        threshold_type="max",
        citation="Basel III: Finalising post-crisis reforms (BCBS 457), December 2017"
    ),

    # CFTC Requirements
    "cftc_audit_trail": RegulatoryRequirement(
        body=RegulatoryBody.CFTC,
        rule_id="17-CFR-1.31",
        requirement_name="Audit Trail Completeness",
        description=(
            "All automated trading decisions must have complete, immutable audit trails "
            "that document the inputs, processing logic, and outputs for each decision."
        ),
        threshold=CFTC_AUDIT_COMPLETENESS_THRESHOLD,
        threshold_type="min",
        citation="CFTC Regulation 1.31 - Books and Records"
    ),

    "cftc_ai_documentation": RegulatoryRequirement(
        body=RegulatoryBody.CFTC,
        rule_id="TAC-2020-AI",
        requirement_name="AI System Documentation",
        description=(
            "AI systems used in trading must document model architecture, training data, "
            "validation procedures, and ongoing monitoring protocols."
        ),
        threshold=None,
        threshold_type="qualitative",
        citation="CFTC Technology Advisory Committee AI Recommendations (2020)"
    ),

    # SEC Requirements
    "sec_citation_accuracy": RegulatoryRequirement(
        body=RegulatoryBody.SEC,
        rule_id="17-CFR-240.10b-5",
        requirement_name="Source Attribution Accuracy",
        description=(
            "AI-generated content citing regulatory filings must accurately reference "
            "the source documents. Fabricated or hallucinated citations violate anti-fraud "
            "provisions."
        ),
        threshold=SEC_CITATION_ACCURACY_THRESHOLD,
        threshold_type="min",
        citation="SEC Rule 10b-5: Employment of Manipulative and Deceptive Practices"
    ),

    "sec_record_retention": RegulatoryRequirement(
        body=RegulatoryBody.SEC,
        rule_id="17-CFR-240.17a-4",
        requirement_name="Record Retention Requirements",
        description=(
            "Broker-dealers must retain records of AI-assisted communications and decisions "
            "in non-rewritable, non-erasable format for specified retention periods."
        ),
        threshold=None,
        threshold_type="qualitative",
        citation="SEC Rule 17a-4: Records to be Preserved by Certain Exchange Members"
    ),

    # GAAP Requirements
    "gaap_materiality": RegulatoryRequirement(
        body=RegulatoryBody.GAAP,
        rule_id="ASC-450-20",
        requirement_name="Materiality Threshold",
        description=(
            "Numerical outputs from AI systems affecting financial statements must be "
            "within 5% of expected values. Deviations exceeding this threshold may require "
            "disclosure or restatement."
        ),
        threshold=GAAP_MATERIALITY_THRESHOLD,
        threshold_type="max",
        citation="FASB ASC 450-20; SEC SAB No. 99 (1999)"
    ),

    # EU AI Act Requirements
    "eu_ai_act_high_risk": RegulatoryRequirement(
        body=RegulatoryBody.EU_AI_ACT,
        rule_id="Art-6-Annex-III",
        requirement_name="High-Risk AI System Requirements",
        description=(
            "AI systems used in creditworthiness assessment or financial risk evaluation "
            "are classified as high-risk and must meet transparency, documentation, and "
            "human oversight requirements."
        ),
        threshold=None,
        threshold_type="qualitative",
        citation="EU AI Act (2024), Article 6, Annex III Category 5(b)"
    ),

    "eu_ai_act_transparency": RegulatoryRequirement(
        body=RegulatoryBody.EU_AI_ACT,
        rule_id="Art-13",
        requirement_name="Transparency and Information",
        description=(
            "High-risk AI systems must be designed to enable users to interpret outputs "
            "and use them appropriately. This includes logging capabilities for traceability."
        ),
        threshold=None,
        threshold_type="qualitative",
        citation="EU AI Act (2024), Article 13: Transparency"
    ),
}


# =============================================================================
# TASK-SPECIFIC REGULATORY MAPPINGS
# =============================================================================

# Maps each task type to the regulatory requirements it must satisfy
TASK_REGULATORY_MAPPINGS: Dict[str, List[str]] = {
    "rag": [
        "fsb_consistent_decisions",      # Identical queries → identical answers
        "sec_citation_accuracy",         # Citations must be valid
        "sec_record_retention",          # Full audit trail required
        "eu_ai_act_transparency",        # Interpretable outputs
    ],
    "sql": [
        "gaap_materiality",              # Numeric results within 5%
        "bis_risk_calculation_tolerance", # Reproducible calculations
        "cftc_audit_trail",              # Complete decision documentation
        "fsb_consistent_decisions",      # Deterministic outputs
    ],
    "summary": [
        "fsb_consistent_decisions",      # Consistent summarization
        "sec_record_retention",          # Audit trail for client communications
        "eu_ai_act_high_risk",           # High-risk system requirements
        "cftc_ai_documentation",         # Model behavior documentation
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
    Validate numeric output against GAAP materiality threshold.

    Per SEC Staff Accounting Bulletin No. 99 and FASB ASC 450-20, a deviation
    is material if it exceeds 5% of the expected value and would influence
    the judgment of a reasonable investor.

    Args:
        actual_value: The value produced by the AI system
        expected_value: The expected/correct value
        threshold: Materiality threshold (default: 5% per GAAP)

    Returns:
        Dict with compliance status and regulatory metadata
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
        "regulatory_body": RegulatoryBody.GAAP.value,
        "rule_citation": "FASB ASC 450-20; SEC SAB No. 99"
    }


def validate_fsb_consistency(
    outputs: List[str],
    require_identity: bool = True
) -> Dict[str, Any]:
    """
    Validate output consistency per FSB BCBS-239 Principle 6.

    FSB requires that identical inputs produce identical outputs for
    regulatory reporting purposes. This ensures reproducibility of
    risk aggregation across systems and time periods.

    Args:
        outputs: List of outputs from identical inputs
        require_identity: If True, requires 100% identity; otherwise reports rate

    Returns:
        Dict with compliance status and consistency metrics
    """
    if not outputs:
        return {
            "compliant": False,
            "error": "No outputs provided for consistency validation"
        }

    reference = outputs[0]
    identical_count = sum(1 for o in outputs if o == reference)
    identity_rate = identical_count / len(outputs)

    is_compliant = identity_rate >= FSB_IDENTITY_REQUIREMENT if require_identity else True

    return {
        "compliant": is_compliant,
        "identity_rate": identity_rate,
        "total_outputs": len(outputs),
        "identical_outputs": identical_count,
        "threshold": FSB_IDENTITY_REQUIREMENT,
        "regulatory_basis": REGULATORY_REQUIREMENTS["fsb_consistent_decisions"],
        "requirement_id": "fsb_consistent_decisions",
        "regulatory_body": RegulatoryBody.FSB.value,
        "rule_citation": "FSB BCBS-239 Principle 6: Accuracy"
    }


def validate_sec_citations(
    citations: List[str],
    available_sources: List[str],
    threshold: float = SEC_CITATION_ACCURACY_THRESHOLD
) -> Dict[str, Any]:
    """
    Validate citation accuracy per SEC Rule 10b-5.

    AI-generated content that cites SEC filings must accurately reference
    actual documents. Fabricated citations may constitute a violation of
    anti-fraud provisions.

    Args:
        citations: List of citations extracted from AI output
        available_sources: List of valid source document identifiers
        threshold: Minimum accuracy threshold (default: 95%)

    Returns:
        Dict with compliance status and citation analysis
    """
    if not citations:
        # No citations is compliant (no false citations)
        return {
            "compliant": True,
            "citation_accuracy": 1.0,
            "total_citations": 0,
            "valid_citations": [],
            "invalid_citations": [],
            "regulatory_basis": REGULATORY_REQUIREMENTS["sec_citation_accuracy"],
            "requirement_id": "sec_citation_accuracy",
            "regulatory_body": RegulatoryBody.SEC.value,
            "rule_citation": "SEC Rule 10b-5"
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
        "regulatory_body": RegulatoryBody.SEC.value,
        "rule_citation": "SEC Rule 10b-5"
    }


def validate_cftc_audit_trail(
    trace_record: Dict[str, Any],
    required_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate audit trail completeness per CFTC Regulation 1.31.

    CFTC requires complete, immutable audit trails for automated trading
    decisions. This validates that all required fields are present in
    the trace record.

    Args:
        trace_record: The audit trail record to validate
        required_fields: List of required field names (defaults to standard set)

    Returns:
        Dict with compliance status and field analysis
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
        "regulatory_body": RegulatoryBody.CFTC.value,
        "rule_citation": "CFTC Regulation 1.31"
    }


# =============================================================================
# COMPOSITE VALIDATION
# =============================================================================

def validate_task_compliance(
    task_type: str,
    validation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Aggregate validation results for a task against all applicable regulations.

    Args:
        task_type: One of 'rag', 'sql', 'summary'
        validation_results: Dict mapping requirement_id to validation result

    Returns:
        Composite compliance report with per-regulation breakdown
    """
    applicable_requirements = TASK_REGULATORY_MAPPINGS.get(task_type, [])

    compliance_results = {}
    all_compliant = True

    for req_id in applicable_requirements:
        if req_id in validation_results:
            result = validation_results[req_id]
            compliance_results[req_id] = result
            if not result.get("compliant", False):
                all_compliant = False
        else:
            compliance_results[req_id] = {
                "compliant": None,
                "status": "not_evaluated",
                "requirement": REGULATORY_REQUIREMENTS.get(req_id)
            }

    return {
        "task_type": task_type,
        "overall_compliant": all_compliant,
        "applicable_requirements": applicable_requirements,
        "validation_results": compliance_results,
        "regulatory_bodies_involved": list(set(
            REGULATORY_REQUIREMENTS[req_id].body.value
            for req_id in applicable_requirements
            if req_id in REGULATORY_REQUIREMENTS
        ))
    }


def get_regulatory_metadata_for_task(task_type: str) -> Dict[str, Any]:
    """
    Get regulatory metadata for inclusion in audit trail entries.

    This provides the regulatory context for each task type, suitable
    for embedding in JSONL trace records.

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
            "compliance_standard": "ACM_ICAIF_2025_Financial_AI"
        }
    }
