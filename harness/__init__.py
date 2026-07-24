#!/usr/bin/env python3
"""
Historical LLM output-drift workshop harness.

This namespace preserves the original workshop API. Its rule mappings and
thresholds are illustrative benchmark controls, not legal guidance, deployment
authorization, or compliance determinations. The installable ``dfah`` package
is the current replay-measurement API.

- DeterministicRetriever: SEC Regulation S-K disclosure precedence encoding
- Task Definitions: configurable benchmark tolerances
- Regulatory Invariants: illustrative control mappings
- Cross-Provider Validation: Multi-provider consistency gates

AI4F Workshop 2025: "LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows"
ICLR 2026 FinAI: "Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness"
"""

__version__ = "1.1.0"  # Historical workshop API
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
    DEFAULT_NUMERIC_TOLERANCE,
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

    # Benchmark constants
    "DEFAULT_NUMERIC_TOLERANCE",
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
