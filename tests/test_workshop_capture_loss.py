"""Behavioral checks for the offline capture-loss exercise."""

import pytest
from scripts.workshop.capture_loss_demo import compare_records, demonstration


def test_repeated_deletions_keep_agreement_but_fail_reference_check():
    views = demonstration()["views"]
    assert views["unchanged"]["reference_mismatches"] == 0
    for name in ("drop_first", "drop_all"):
        assert views[name]["recorded_path_agreement"] == 1.0
        assert views[name]["reference_mismatches"] == 3


def test_missing_record_is_unknown_but_observed_empty_is_comparable():
    result = demonstration()
    assert result["views"]["missing_record"]["recorded_path_agreement"] is None
    assert result["views"]["missing_record"]["missing_records"] == 1
    assert result["observed_empty_reference"]["recorded_path_agreement"] == 1.0
    assert result["observed_empty_reference"]["reference_mismatches"] == 0


def test_reference_check_detects_changed_arguments_with_matching_tool_names():
    reference = [[{"name": "read_balance", "arguments": {"account": "A"}}]] * 3
    recorded = [[{"name": "read_balance", "arguments": {"account": "B"}}]] * 3
    result = compare_records(recorded, reference)
    assert result["recorded_path_agreement"] == 1.0
    assert result["reference_mismatches"] == 3


@pytest.mark.parametrize("recorded,reference", [([], []), ([[]], [[], []])])
def test_denominator_must_be_nonempty_and_paired(recorded, reference):
    with pytest.raises(ValueError, match="same nonzero denominator"):
        compare_records(recorded, reference)
