#!/usr/bin/env python3
"""
Task definitions for financial LLM evaluation with regulatory compliance invariants.

This module implements finance-calibrated validation logic for LLM outputs in
regulated financial workflows. Each validation function is designed to satisfy
specific regulatory requirements, not generic ML evaluation.

Includes:
- RAG Q&A over SEC filings with citation validation (SEC Rule 10b-5 compliance)
- Policy-bounded JSON summarization with schema constraints (FSB consistency)
- Text-to-SQL with GAAP materiality invariant checking (ASC 450-20)

Regulatory Framework:
    - GAAP ASC 450-20: 5% materiality threshold for financial disclosures
    - SEC Rule 10b-5: Anti-fraud provisions requiring accurate source attribution
    - FSB BCBS-239: Consistent decision outputs for regulatory reporting
    - CFTC Rule 17a-4: Audit trail requirements for automated decisions

ACM ICAIF 2025: "LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows"
"""
import re
import json
from typing import List, Dict, Tuple, Any
from jsonschema import validate, ValidationError


# =============================================================================
# REGULATORY COMPLIANCE CONSTANTS
# =============================================================================

# GAAP Materiality Threshold (ASC 450-20 / SEC SAB No. 99)
# Per SEC Staff Accounting Bulletin No. 99, a 5% deviation from expected values
# is the de facto threshold for determining materiality in financial statements.
# This threshold is applied to SQL query results affecting financial calculations.
GAAP_MATERIALITY_THRESHOLD: float = 0.05  # 5% tolerance

# SEC Citation Accuracy Requirement (Rule 10b-5)
# AI systems generating content with SEC filing citations must achieve high
# accuracy to avoid anti-fraud violations. Hallucinated citations are prohibited.
SEC_CITATION_ACCURACY_MINIMUM: float = 0.95  # 95% minimum citation validity

# FSB Consistency Requirement (BCBS-239 Principle 6)
# Identical inputs must produce identical outputs for regulatory reporting.
# This is stricter than generic ML reproducibility requirements.
FSB_IDENTITY_RATE_TARGET: float = 1.0  # 100% identity at T=0.0


# JSON schema for policy-bounded summarization
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "client_name": {"type": "string"},
        "summary": {"type": "string"},
        "compliance_disclaimer": {
            "type": "string",
            "enum": ["This is not investment advice."]
        },
    },
    "required": ["client_name", "summary", "compliance_disclaimer"],
    "additionalProperties": False
}


def format_rag_prompt(question: str, snippets: List[Tuple[str, str, Dict[str, Any]]]) -> List[Dict[str, str]]:
    """
    Format RAG prompt with proper citation instructions for SEC filings.

    Args:
        question: User question
        snippets: Retrieved snippets as (snippet_id, text, metadata) tuples

    Returns:
        Formatted messages for LLM [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    # Format context with source identifiers
    context_parts = []
    for snippet_id, text, meta in snippets:
        # Use base source name for citations (e.g., "citi_2024_10k")
        source_name = snippet_id.split('#')[0]
        context_parts.append(f"[{source_name}] {text}")

    context = "\n\n".join(context_parts)

    system_msg = (
        "You are a precise financial analyst. Answer the question using only the provided documents. "
        "CITE sources in square brackets using the file base name, e.g., [citi_2024_10k]. "
        "Only cite documents you actually reference in your answer."
    )

    user_msg = f"Question: {question}\n\nDocuments:\n{context}\n\nAnswer with citations:"

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]


def format_summary_prompt(profile_text: str) -> List[Dict[str, str]]:
    """
    Format policy-bounded JSON summarization prompt.

    Enforces:
    - Fixed schema with required fields
    - Exact compliance disclaimer text
    - Structured output format

    Args:
        profile_text: Client profile description

    Returns:
        Formatted messages for LLM
    """
    system_msg = (
        "You produce STRICT JSON with keys: client_name, summary, compliance_disclaimer. "
        'The disclaimer MUST be exactly: "This is not investment advice." '
        "Return ONLY valid JSON, no additional text."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": profile_text}
    ]


def format_sql_prompt(question: str, schema_desc: str = None) -> List[Dict[str, str]]:
    """
    Format text-to-SQL prompt with schema constraints.

    Args:
        question: Natural language query
        schema_desc: Optional custom schema description

    Returns:
        Formatted messages for LLM
    """
    if schema_desc is None:
        schema_desc = (
            "Schema: transactions(id INT, date TEXT, region TEXT, amount REAL, category TEXT). "
            "Use double quotes for strings."
        )

    system_msg = (
        f"You write SQLite SQL ONLY. No prose, no explanations. {schema_desc} "
        "Return ONLY the SQL query, nothing else."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question}
    ]


def extract_citations(text: str) -> List[str]:
    """
    Extract citations from LLM output.

    Supports both formats:
    - New format: [citi_2024_10k]
    - Legacy format: [CITATION:citi_2024_10k]

    Args:
        text: LLM response text

    Returns:
        Sorted list of cited source names
    """
    citations = set()

    # Legacy format: [CITATION:name]
    legacy_pattern = re.compile(r"\[CITATION:([^\]]+)\]")
    for match in legacy_pattern.finditer(text):
        citations.add(match.group(1))

    # New format: [name] (but exclude things like [100] or [CITATION:...])
    new_pattern = re.compile(r"\[([A-Za-z0-9._-]+)\]")
    for match in new_pattern.finditer(text):
        cite = match.group(1)
        # Exclude numeric-only citations and CITATION: prefix
        if not cite.startswith("CITATION:") and not cite.isdigit():
            citations.add(cite)

    return sorted(list(citations))


def validate_citations(
    citations: List[str],
    available_sources: List[str],
    sec_accuracy_threshold: float = SEC_CITATION_ACCURACY_MINIMUM
) -> Dict[str, Any]:
    """
    Validate citation accuracy per SEC Rule 10b-5 anti-fraud requirements.

    AI systems generating content with SEC filing citations must accurately
    reference actual provided documents. Fabricated or "hallucinated" citations
    may constitute a violation of anti-fraud provisions under SEC Rule 10b-5.

    Regulatory Basis:
        - SEC Rule 10b-5: Prohibition against manipulative and deceptive practices
        - SEC Rule 17a-4: Record retention requiring accurate source attribution
        - FSB Principles: Traceability requirements for AI-generated financial content

    The citation accuracy threshold (95%) is stricter than generic RAG evaluation
    because false citations in financial contexts have regulatory consequences.

    Args:
        citations: List of cited sources extracted from AI output
        available_sources: List of valid source document identifiers
        sec_accuracy_threshold: Minimum accuracy for SEC compliance (default: 95%)

    Returns:
        {
            "valid_citations": List[str],
            "invalid_citations": List[str],
            "citation_accuracy": float,  # 0.0-1.0
            "sec_compliant": bool,  # Meets SEC accuracy threshold
            "regulatory_threshold": float,
            "regulatory_basis": str
        }
    """
    # Normalize available sources (handle with/without .txt)
    normalized_sources = set()
    for source in available_sources:
        normalized_sources.add(source)
        if source.endswith('.txt'):
            normalized_sources.add(source[:-4])
        else:
            normalized_sources.add(source + '.txt')

    valid_citations = []
    invalid_citations = []

    for cite in citations:
        if cite in normalized_sources:
            valid_citations.append(cite)
        else:
            invalid_citations.append(cite)

    # Calculate citation accuracy
    citation_accuracy = len(valid_citations) / len(citations) if citations else 1.0

    # SEC compliance check: accuracy must meet or exceed threshold
    sec_compliant = citation_accuracy >= sec_accuracy_threshold

    return {
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "citation_accuracy": citation_accuracy,
        "sec_compliant": sec_compliant,
        "regulatory_threshold": sec_accuracy_threshold,
        "regulatory_basis": "SEC Rule 10b-5; SEC Rule 17a-4"
    }


def validate_summary_json(output: str) -> Dict[str, Any]:
    """
    Validate JSON summary against schema.

    Args:
        output: LLM JSON output

    Returns:
        {
            "valid": bool,
            "parsed": dict|None,
            "error": str|None
        }
    """
    try:
        parsed = json.loads(output)
        validate(parsed, SUMMARY_SCHEMA)
        return {"valid": True, "parsed": parsed, "error": None}
    except json.JSONDecodeError as e:
        return {"valid": False, "parsed": None, "error": f"JSON decode error: {e}"}
    except ValidationError as e:
        return {"valid": False, "parsed": None, "error": f"Schema validation error: {e.message}"}


def validate_sql_query(
    sql: str,
    connection,
    expected_total: float = None,
    gaap_materiality_threshold: float = GAAP_MATERIALITY_THRESHOLD
) -> Dict[str, Any]:
    """
    Validate SQL query execution against GAAP materiality invariants.

    This validation implements the GAAP ASC 450-20 materiality threshold for
    financial calculations. Per SEC Staff Accounting Bulletin No. 99, deviations
    exceeding 5% of expected values are considered material and may require
    disclosure or indicate a compliance failure.

    Regulatory Basis:
        - GAAP ASC 450-20: Loss contingencies and materiality assessment
        - SEC SAB No. 99: Quantitative threshold of 5% for materiality
        - BIS BCBS-457: Reproducibility requirements for risk calculations

    The tolerance is NOT a generic ML hyperparameter—it is a finance-calibrated
    threshold derived from regulatory auditing standards.

    Args:
        sql: SQL query to validate
        connection: SQLite database connection
        expected_total: Expected total for SUM queries (regulatory baseline)
        gaap_materiality_threshold: GAAP materiality threshold (default: 5% per ASC 450-20)

    Returns:
        {
            "executable": bool,
            "gaap_compliant": bool,  # Renamed from decision_ok for regulatory clarity
            "result": Any,
            "error": str|None,
            "materiality_deviation": float|None,  # Actual deviation from expected
            "regulatory_threshold": float,  # Applied GAAP threshold
            "regulatory_basis": str  # Citation to regulatory requirement
        }
    """
    import pandas as pd

    sql_clean = sql.strip().strip("`").strip()

    # Guard against non-SELECT statements from LLM-generated SQL
    _sql_upper = sql_clean.upper().lstrip()
    _FORBIDDEN = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
                  "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA")
    if not _sql_upper.startswith("SELECT") and not _sql_upper.startswith("WITH"):
        return {
            "executable": False,
            "gaap_compliant": False,
            "decision_ok": False,
            "result": None,
            "error": f"Only SELECT/WITH queries are permitted, got: {_sql_upper[:20]}",
            "materiality_deviation": None,
            "regulatory_threshold": gaap_materiality_threshold,
            "regulatory_basis": "GAAP ASC 450-20; SEC SAB No. 99"
        }
    for keyword in _FORBIDDEN:
        if keyword in _sql_upper:
            return {
                "executable": False,
                "gaap_compliant": False,
                "decision_ok": False,
                "result": None,
                "error": f"Forbidden SQL keyword detected: {keyword}",
                "materiality_deviation": None,
                "regulatory_threshold": gaap_materiality_threshold,
                "regulatory_basis": "GAAP ASC 450-20; SEC SAB No. 99"
            }

    try:
        df = pd.read_sql_query(sql_clean, connection)

        # Check if query returns results
        if len(df) == 0:
            return {
                "executable": True,
                "gaap_compliant": False,
                "decision_ok": False,  # Backward compatibility
                "result": df,
                "error": "Query returned no results",
                "materiality_deviation": None,
                "regulatory_threshold": gaap_materiality_threshold,
                "regulatory_basis": "GAAP ASC 450-20; SEC SAB No. 99"
            }

        # Validate SUM queries against expected total using GAAP materiality threshold
        gaap_compliant = True
        materiality_deviation = None

        if expected_total is not None and "sum(" in sql_clean.lower() and "amount" in sql_clean.lower():
            actual_value = float(df.iloc[0, 0]) if len(df) and len(df.columns) else float("nan")

            # Calculate materiality deviation as percentage of expected value
            if expected_total != 0:
                materiality_deviation = abs(actual_value - expected_total) / abs(expected_total)
            else:
                materiality_deviation = float('inf') if actual_value != 0 else 0.0

            # GAAP compliance: deviation must be within materiality threshold
            gaap_compliant = materiality_deviation <= gaap_materiality_threshold

        return {
            "executable": True,
            "gaap_compliant": gaap_compliant,
            "decision_ok": gaap_compliant,  # Backward compatibility alias
            "result": df,
            "error": None,
            "materiality_deviation": materiality_deviation,
            "regulatory_threshold": gaap_materiality_threshold,
            "regulatory_basis": "GAAP ASC 450-20; SEC SAB No. 99"
        }

    except Exception as e:
        return {
            "executable": False,
            "gaap_compliant": False,
            "decision_ok": False,  # Backward compatibility
            "result": None,
            "error": str(e),
            "materiality_deviation": None,
            "regulatory_threshold": gaap_materiality_threshold,
            "regulatory_basis": "GAAP ASC 450-20; SEC SAB No. 99"
        }
