#!/usr/bin/env python3
"""
Historical task definitions for the financial LLM workshop.

The checks below are reproducible benchmark rules, not legal interpretations or
compliance determinations. Names retained for backward compatibility are noted
where they overstate the underlying measurement.

Includes:
- RAG Q&A over SEC filings with source-reference matching
- Policy-bounded JSON summarization with schema constraints
- Text-to-SQL with a configurable numeric tolerance

Any mapping from these measurements to a legal or policy obligation must be
defined and approved for the intended workflow.

AI4F Workshop 2025: "LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows"
"""
import re
import json
from typing import List, Dict, Tuple, Any
from jsonschema import validate, ValidationError


# =============================================================================
# HISTORICAL WORKSHOP DEFAULTS
# =============================================================================

# The workshop used 5% as an illustrative numeric comparison tolerance. There is
# no universal 5% GAAP materiality threshold; production values must come from
# an approved task contract.
DEFAULT_NUMERIC_TOLERANCE: float = 0.05
# Deprecated compatibility alias. Do not interpret this name as legal guidance.
GAAP_MATERIALITY_THRESHOLD: float = DEFAULT_NUMERIC_TOLERANCE

# Historical workshop source-reference target. This checks whether extracted
# identifiers occur in the supplied source list; it is not a legal threshold.
SEC_CITATION_ACCURACY_MINIMUM: float = 0.95  # 95% minimum citation validity

# Historical workshop exact-output target. BCBS 239 does not prescribe
# identical LLM outputs.
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
    Compare extracted citation identifiers with the supplied source list.

    The 95% default is a historical workshop setting. Identifier matching does
    not establish factual grounding, legal compliance, or whether a record must
    be retained.

    Args:
        citations: List of cited sources extracted from AI output
        available_sources: List of valid source document identifiers
        sec_accuracy_threshold: Configured source-match threshold (default: 95%)

    Returns:
        {
            "valid_citations": List[str],
            "invalid_citations": List[str],
            "citation_accuracy": float,  # 0.0-1.0
            "sec_compliant": bool,  # Legacy alias for passed_profile
            "regulatory_threshold": float,
            "regulatory_basis": str,
            "interpretation": str
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

    # Historical compatibility field: whether the source-match profile passed.
    sec_compliant = citation_accuracy >= sec_accuracy_threshold

    return {
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "citation_accuracy": citation_accuracy,
        "sec_compliant": sec_compliant,
        "passed_profile": sec_compliant,
        "regulatory_threshold": sec_accuracy_threshold,
        "regulatory_basis": "Historical workshop source-reference profile",
        "interpretation": (
            "Illustrative identifier-match result; not factual, legal, or "
            "regulatory compliance"
        ),
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
    Execute a SQL query and apply the workshop's configurable numeric tolerance.

    The ``gaap_materiality_threshold`` parameter and ``gaap_compliant`` result
    key are legacy API names. They indicate whether a numeric result falls
    within the configured tolerance; they do not establish GAAP materiality or
    regulatory compliance.

    Args:
        sql: SQL query to validate
        connection: SQLite database connection
        expected_total: Expected total for SUM queries
        gaap_materiality_threshold: Legacy-named numeric tolerance (default: 5% for the exercise)

    Returns:
        {
            "executable": bool,
            "gaap_compliant": bool,  # Legacy alias for within_tolerance
            "result": Any,
            "error": str|None,
            "materiality_deviation": float|None,  # Legacy name for numeric deviation
            "regulatory_threshold": float,  # Legacy name for configured tolerance
            "regulatory_basis": str  # Legacy metadata; not a legal determination
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
            "regulatory_basis": "Illustrative task tolerance; configure per approved workflow"
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
                "regulatory_basis": "Illustrative task tolerance; configure per approved workflow"
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
                "regulatory_basis": "Illustrative task tolerance; configure per approved workflow"
            }

        # Validate SUM queries against the configured numeric tolerance.
        gaap_compliant = True
        materiality_deviation = None

        if expected_total is not None and "sum(" in sql_clean.lower() and "amount" in sql_clean.lower():
            actual_value = float(df.iloc[0, 0]) if len(df) and len(df.columns) else float("nan")

            # Calculate numeric deviation as a percentage of the expected value.
            if expected_total != 0:
                materiality_deviation = abs(actual_value - expected_total) / abs(expected_total)
            else:
                materiality_deviation = float('inf') if actual_value != 0 else 0.0

            # Legacy variable name: this is a tolerance check, not GAAP compliance.
            gaap_compliant = materiality_deviation <= gaap_materiality_threshold

        return {
            "executable": True,
            "gaap_compliant": gaap_compliant,
            "decision_ok": gaap_compliant,  # Backward compatibility alias
            "result": df,
            "error": None,
            "materiality_deviation": materiality_deviation,
            "regulatory_threshold": gaap_materiality_threshold,
            "regulatory_basis": "Illustrative task tolerance; configure per approved workflow"
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
            "regulatory_basis": "Illustrative task tolerance; configure per approved workflow"
        }
