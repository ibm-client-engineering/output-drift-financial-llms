#!/usr/bin/env python3
"""
LLM Output Drift Evaluation Harness

A deterministic evaluation framework for financial AI compliance.
"""

__version__ = "1.0.0"
__author__ = "Raffi Khatchadourian, Rolando Franco"
__license__ = "Apache 2.0"

from .deterministic_retriever import DeterministicRetriever
from .task_definitions import (
    format_rag_prompt,
    format_summary_prompt,
    format_sql_prompt,
    extract_citations,
    validate_citations
)
from .cross_provider_validation import CrossProviderValidator

__all__ = [
    "DeterministicRetriever",
    "format_rag_prompt",
    "format_summary_prompt",
    "format_sql_prompt",
    "extract_citations",
    "validate_citations",
    "CrossProviderValidator"
]
