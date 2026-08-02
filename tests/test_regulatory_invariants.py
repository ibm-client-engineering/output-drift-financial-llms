"""Regression tests for fail-closed historical profile validators."""

from typing import Any

import pytest
from harness.regulatory_invariants import (
    TASK_REGULATORY_MAPPINGS,
    validate_fsb_consistency,
    validate_task_compliance,
)


@pytest.mark.parametrize(
    "outputs",
    [
        [],
        [None, None],
        ["", ""],
        ["  ", "\t"],
        ["escalate", None],
        [1, 1],
        ["escalate", {"decision": "escalate"}],
    ],
)
def test_invalid_consistency_observations_fail_closed(outputs: list[Any]) -> None:
    result = validate_fsb_consistency(outputs)
    informational = validate_fsb_consistency(outputs, require_identity=False)

    assert result["compliant"] is False
    assert result["passed_profile"] is False
    assert result["identity_rate"] is None
    assert result["total_outputs"] == len(outputs)
    assert result["status"] in {"invalid_input", "not_evaluated"}

    if outputs:
        assert result["invalid_output_count"] >= 1
        assert result["invalid_output_indices"]

    assert informational["compliant"] is False
    assert informational["passed_profile"] is False


def test_valid_consistency_observations_preserve_profile_behavior() -> None:
    identical = validate_fsb_consistency(["escalate", "escalate"])
    different = validate_fsb_consistency(["escalate", "dismiss"])
    informational = validate_fsb_consistency(
        ["escalate", "dismiss"],
        require_identity=False,
    )

    assert identical["compliant"] is True
    assert identical["passed_profile"] is True
    assert identical["identity_rate"] == 1.0
    assert identical["status"] == "passed"

    assert different["compliant"] is False
    assert different["passed_profile"] is False
    assert different["identity_rate"] == 0.5
    assert different["status"] == "failed"

    assert informational["compliant"] is True
    assert informational["identity_rate"] == 0.5
    assert informational["status"] == "passed"


def test_unknown_task_type_fails_closed() -> None:
    result = validate_task_compliance("unknown_task", {})

    assert result["overall_compliant"] is False
    assert result["all_profiles_passed"] is False
    assert result["status"] == "invalid_task_type"
    assert result["applicable_requirements"] == []
    assert result["validation_results"] == {}
    assert result["supported_task_types"] == ["rag", "sql", "summary"]


@pytest.mark.parametrize("task_type", sorted(TASK_REGULATORY_MAPPINGS))
def test_missing_required_profiles_are_incomplete(task_type: str) -> None:
    result = validate_task_compliance(task_type, {})

    assert result["overall_compliant"] is False
    assert result["all_profiles_passed"] is False
    assert result["status"] == "incomplete"
    assert result["not_evaluated_requirements"] == TASK_REGULATORY_MAPPINGS[task_type]
    assert all(
        profile["compliant"] is None and profile["status"] == "not_evaluated"
        for profile in result["validation_results"].values()
    )


@pytest.mark.parametrize("task_type", sorted(TASK_REGULATORY_MAPPINGS))
def test_partial_required_profiles_are_incomplete(task_type: str) -> None:
    first_requirement, *missing_requirements = TASK_REGULATORY_MAPPINGS[task_type]

    result = validate_task_compliance(
        task_type,
        {first_requirement: {"compliant": True}},
    )

    assert result["overall_compliant"] is False
    assert result["all_profiles_passed"] is False
    assert result["status"] == "incomplete"
    assert result["not_evaluated_requirements"] == missing_requirements


def test_explicit_not_evaluated_profile_is_incomplete() -> None:
    profiles = {
        requirement_id: {"compliant": True}
        for requirement_id in TASK_REGULATORY_MAPPINGS["summary"]
    }
    missing_id = TASK_REGULATORY_MAPPINGS["summary"][0]
    profiles[missing_id] = {"compliant": None, "status": "not_evaluated"}

    result = validate_task_compliance("summary", profiles)

    assert result["overall_compliant"] is False
    assert result["all_profiles_passed"] is False
    assert result["status"] == "incomplete"
    assert result["not_evaluated_requirements"] == [missing_id]


def test_not_evaluated_status_overrides_contradictory_boolean() -> None:
    profiles = {
        requirement_id: {"compliant": True}
        for requirement_id in TASK_REGULATORY_MAPPINGS["summary"]
    }
    missing_id = TASK_REGULATORY_MAPPINGS["summary"][0]
    profiles[missing_id] = {"compliant": True, "status": "not_evaluated"}

    result = validate_task_compliance("summary", profiles)

    assert result["overall_compliant"] is False
    assert result["all_profiles_passed"] is False
    assert result["status"] == "incomplete"
    assert result["not_evaluated_requirements"] == [missing_id]


def test_complete_required_profiles_preserve_aggregate_behavior() -> None:
    passing_profiles = {
        requirement_id: {"compliant": True}
        for requirement_id in TASK_REGULATORY_MAPPINGS["sql"]
    }
    passed = validate_task_compliance("sql", passing_profiles)

    assert passed["overall_compliant"] is True
    assert passed["all_profiles_passed"] is True
    assert passed["status"] == "passed"
    assert passed["not_evaluated_requirements"] == []

    failing_id = TASK_REGULATORY_MAPPINGS["sql"][0]
    passing_profiles[failing_id] = {"compliant": False}
    failed = validate_task_compliance("sql", passing_profiles)

    assert failed["overall_compliant"] is False
    assert failed["all_profiles_passed"] is False
    assert failed["status"] == "failed"
    assert failed["not_evaluated_requirements"] == []
