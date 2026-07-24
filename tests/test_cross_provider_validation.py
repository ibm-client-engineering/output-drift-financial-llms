"""Regression tests for conservative cross-provider comparison semantics."""

from harness.cross_provider_validation import CrossProviderValidator


def test_rag_citation_drift_fails_overall_consistency() -> None:
    validator = CrossProviderValidator(providers=["a", "b"])
    outputs = {"a": "same answer", "b": "same answer"}
    citations = {"a": ["source-a"], "b": ["source-b"]}

    result = validator.validate(outputs, task_type="rag", citations=citations)

    assert result["consistent"] is False
    assert result["task_validation"]["citation_consistent"] is False


def test_rag_missing_citation_channel_fails_closed() -> None:
    validator = CrossProviderValidator(providers=["a", "b"])
    outputs = {"a": "same answer", "b": "same answer"}

    result = validator.validate(outputs, task_type="rag")

    assert result["consistent"] is False
    assert result["task_validation"]["citation_coverage_complete"] is False


def test_summary_schema_drift_fails_overall_consistency() -> None:
    validator = CrossProviderValidator(providers=["a", "b"])
    outputs = {
        "a": '{"decision": "approve"}',
        "b": '{"outcome": "approve"}',
    }

    result = validator.validate(outputs, task_type="summary")

    assert result["consistent"] is False
    assert result["task_validation"]["schema_consistent"] is False


def test_summary_invalid_json_fails_closed() -> None:
    validator = CrossProviderValidator(providers=["a", "b"])
    outputs = {"a": "not-json", "b": "not-json"}

    result = validator.validate(outputs, task_type="summary")

    assert result["consistent"] is False
    assert result["task_validation"]["schema_coverage_complete"] is False


def test_sql_requires_executed_results_for_equivalence() -> None:
    validator = CrossProviderValidator(providers=["a", "b"])
    queries = {
        "a": "SELECT SUM(amount) FROM transactions",
        "b": "SELECT SUM(amount) FROM transactions",
    }

    result = validator.validate(queries, task_type="sql")

    assert result["consistent"] is False
    assert result["task_validation"]["result_coverage_complete"] is False
    assert result["task_validation"]["result_match"]["b"] is None


def test_sql_passes_when_executed_results_match() -> None:
    validator = CrossProviderValidator(providers=["a", "b"])
    queries = {
        "a": "SELECT SUM(amount) FROM transactions",
        "b": "SELECT SUM(amount) FROM transactions",
    }
    results = {"a": 100.0, "b": 100.0}

    result = validator.validate(queries, task_type="sql", sql_results=results)

    assert result["consistent"] is True
    assert result["task_validation"]["result_coverage_complete"] is True
    assert result["task_validation"]["result_match"]["b"] is True


def test_sql_numeric_tolerance_is_provider_order_invariant() -> None:
    validator = CrossProviderValidator(providers=["a", "b"], tolerance_pct=5.0)
    queries = {"a": "SELECT 1", "b": "SELECT 1"}

    forward = validator.validate(
        queries, task_type="sql", sql_results={"a": 100.0, "b": 105.2}
    )
    reverse = validator.validate(
        {"b": "SELECT 1", "a": "SELECT 1"},
        task_type="sql",
        sql_results={"b": 105.2, "a": 100.0},
    )

    assert forward["consistent"] is reverse["consistent"]
    assert forward["consistent"] is True
