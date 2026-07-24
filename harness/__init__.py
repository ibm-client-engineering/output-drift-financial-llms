#!/usr/bin/env python3
"""
LLM Output Drift Evaluation Harness for Financial AI Compliance.

This harness implements finance-calibrated validation with explicit regulatory
requirement mappings. Each component satisfies specific compliance requirements:

- DeterministicRetriever: SEC Regulation S-K disclosure precedence encoding
- Task Definitions: GAAP ASC 450-20 materiality thresholds
- Regulatory Invariants: FSB/BIS/CFTC/SEC requirement database
- Cross-Provider Validation: Multi-provider consistency gates

Regulatory Framework:
    - FSB BCBS-239: Consistent decision outputs
    - CFTC Rule 17a-4: Audit trail requirements
    - SEC Rule 10b-5: Citation accuracy requirements
    - GAAP ASC 450-20: 5% materiality threshold

AI4F Workshop 2025: "LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows"
ICLR 2026 FinAI: "Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness"
"""

__version__ = "1.1.0"  # Regulatory compliance refactor
__author__ = "Raffi Khatchadourian, Rolando Franco"
__license__ = "Apache 2.0"

from .deterministic_retriever import (
    DeterministicRetriever,
    SEC_10K_SECTION_PRECEDENCE,
    FSB_IDENTITY_REQUIREMENT
)
from .task_definitions import (
    format_rag_prompt,
    format_summary_prompt,
    format_sql_prompt,
    extract_citations,
    validate_citations,
    validate_sql_query,
    GAAP_MATERIALITY_THRESHOLD,
    SEC_CITATION_ACCURACY_MINIMUM,
    FSB_IDENTITY_RATE_TARGET
)
from .cross_provider_validation import CrossProviderValidator
from .regulatory_invariants import (
    RegulatoryBody,
    RegulatoryRequirement,
    REGULATORY_REQUIREMENTS,
    TASK_REGULATORY_MAPPINGS,
    validate_gaap_materiality,
    validate_fsb_consistency,
    validate_sec_citations,
    validate_cftc_audit_trail,
    validate_task_compliance,
    get_regulatory_metadata_for_task
)
from .compliance_judge import (
    ComplianceJudge,
    ComplianceQuadrant,
    JudgeEvaluation,
    ComplianceAttestation,
    create_ollama_judge,
    create_watsonx_judge
)

__all__ = [
    # Core retrieval
    "DeterministicRetriever",
    "SEC_10K_SECTION_PRECEDENCE",

    # Task formatting
    "format_rag_prompt",
    "format_summary_prompt",
    "format_sql_prompt",
    "extract_citations",
    "validate_citations",
    "validate_sql_query",

    # Regulatory constants
    "GAAP_MATERIALITY_THRESHOLD",
    "SEC_CITATION_ACCURACY_MINIMUM",
    "FSB_IDENTITY_RATE_TARGET",
    "FSB_IDENTITY_REQUIREMENT",

    # Cross-provider validation
    "CrossProviderValidator",

    # Regulatory invariants module
    "RegulatoryBody",
    "RegulatoryRequirement",
    "REGULATORY_REQUIREMENTS",
    "TASK_REGULATORY_MAPPINGS",
    "validate_gaap_materiality",
    "validate_fsb_consistency",
    "validate_sec_citations",
    "validate_cftc_audit_trail",
    "validate_task_compliance",
    "get_regulatory_metadata_for_task",

    # LLM-as-Judge compliance evaluation
    "ComplianceJudge",
    "ComplianceQuadrant",
    "JudgeEvaluation",
    "ComplianceAttestation",
    "create_ollama_judge",
    "create_watsonx_judge"
]
