#!/usr/bin/env python3
"""
Task definitions for financial LLM evaluation.

Includes:
- RAG Q&A over SEC filings with citation validation
- Policy-bounded JSON summarization with schema constraints
- Text-to-SQL with invariant checking
"""
import re
import json
from typing import List, Dict, Tuple, Any
from jsonschema import validate, ValidationError


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


def validate_citations(citations: List[str], available_sources: List[str]) -> Dict[str, Any]:
    """
    Validate that citations reference actual sources.

    Compliance requirement: All citations must map to provided documents.

    Args:
        citations: List of cited sources
        available_sources: List of available source names

    Returns:
        {
            "valid_citations": List[str],
            "invalid_citations": List[str],
            "citation_accuracy": float  # 0.0-1.0
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

    return {
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "citation_accuracy": len(valid_citations) / len(citations) if citations else 1.0
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


def validate_sql_query(sql: str, connection, expected_total: float = None, tolerance_pct: float = 5.0) -> Dict[str, Any]:
    """
    Validate SQL query execution and invariants.

    Finance-calibrated tolerance: ±5% (GAAP materiality threshold).

    Args:
        sql: SQL query to validate
        connection: SQLite database connection
        expected_total: Expected total for SUM queries (if applicable)
        tolerance_pct: Tolerance percentage for numeric validation

    Returns:
        {
            "executable": bool,
            "decision_ok": bool,
            "result": Any,
            "error": str|None
        }
    """
    import pandas as pd

    sql_clean = sql.strip().strip("`").strip()

    try:
        df = pd.read_sql_query(sql_clean, connection)

        # Check if query returns results
        if len(df) == 0:
            return {
                "executable": True,
                "decision_ok": False,
                "result": df,
                "error": "Query returned no results"
            }

        # Validate SUM queries against expected total (if provided)
        decision_ok = True
        if expected_total is not None and "sum(" in sql_clean.lower() and "amount" in sql_clean.lower():
            actual_value = float(df.iloc[0, 0]) if len(df) and len(df.columns) else float("nan")
            tolerance = (tolerance_pct / 100.0) * expected_total
            decision_ok = abs(actual_value - expected_total) <= tolerance

        return {
            "executable": True,
            "decision_ok": decision_ok,
            "result": df,
            "error": None
        }

    except Exception as e:
        return {
            "executable": False,
            "decision_ok": False,
            "result": None,
            "error": str(e)
        }
