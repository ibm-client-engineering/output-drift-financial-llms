from __future__ import annotations

import pytest
from pydantic import ValidationError

import dfah
from dfah import (
    CaseReport,
    ChannelState,
    Eligibility,
    EligibilityReasonCount,
    ParseProvenance,
    Suite,
    TARReport,
    ToolCall,
    Trajectory,
    build_manifest,
)
from dfah.metrics import dar, first_anchored_agreement, modal_agreement, tar
from dfah.parse import parse_decision


def call(name: str, **arguments):
    return ToolCall(
        name=name,
        arguments=arguments,
        output_hash="a" * 64,
        result_state=ChannelState.OBSERVED_NONEMPTY,
    )


def path(*calls: ToolCall) -> Trajectory:
    return Trajectory(
        state=ChannelState.OBSERVED_NONEMPTY if calls else ChannelState.OBSERVED_EMPTY,
        tool_calls=calls,
    )


def test_fail_closed_parser_never_substitutes_a_label():
    malformed = [
        "escalate",
        "DECISION: ESCALATE\nDECISION: DISMISS",
        "DECISION: MADE_UP",
        "I would probably escalate.",
        "",
    ]
    for text in malformed:
        decision, provenance = parse_decision(text, ("escalate", "dismiss", "investigate"))
        assert decision is None
        assert provenance == ParseProvenance(
            strategy="none", raw_span=None, confidence=None, fallback=True, accepted=False
        )


def test_strict_parser_accepts_one_line_marker():
    decision, provenance = parse_decision(
        "Evidence considered.\nDECISION: ESCALATE", ("escalate", "dismiss", "investigate")
    )
    assert decision is not None and decision.label == "escalate"
    assert provenance.accepted and not provenance.fallback


def test_observed_empty_is_not_unavailable():
    assert path().state is ChannelState.OBSERVED_EMPTY
    assert path().state.observed
    assert not ChannelState.UNAVAILABLE.observed
    with pytest.raises(ValidationError):
        Trajectory(state=ChannelState.UNAVAILABLE, tool_calls=(call("x"),))


def test_eligibility_reason_counts_reject_content_bearing_text():
    with pytest.raises(ValidationError):
        EligibilityReasonCount(reason="provider said account 123 failed", groups=1)


def test_shared_denominator_fails_closed():
    eligibility = Eligibility(
        eligible=True,
        decision=ChannelState.OBSERVED_NONEMPTY,
        trajectory=ChannelState.OBSERVED_NONEMPTY,
    )
    with pytest.raises(Exception, match="identical retained denominator"):
        CaseReport(
            case_id="x",
            task="t",
            replay_count=3,
            decision_denominator=3,
            trajectory_denominator=2,
            dar=1,
            tar=TARReport(seq=1, bag=1, set=1, strong=1),
            gap=0,
            eligibility=eligibility,
        )


def test_agreement_coarsening_and_anchor_properties():
    paths = [
        path(call("a", entity="x"), call("b", entity="x")),
        path(call("b", entity="x"), call("a", entity="x")),
        path(call("a", entity="y"), call("b", entity="x")),
    ]
    assert tar(paths, "set") >= tar(paths, "bag") >= tar(paths, "seq")
    assert tar(paths, "strong") <= tar(paths, "seq")
    values = ["b", "a", "a", "c"]
    assert modal_agreement(values).value >= first_anchored_agreement(values)
    assert 1 / len(values) <= dar(values) <= 1


def test_ties_are_explicit_not_insertion_order_modes():
    result = modal_agreement(["b", "a"])
    assert result.tied
    assert result.modes == ("a", "b")


def test_path_variation_triage_distinguishes_arguments_from_sequence():
    eligibility = Eligibility(
        eligible=True,
        decision=ChannelState.OBSERVED_NONEMPTY,
        trajectory=ChannelState.OBSERVED_NONEMPTY,
    )
    report = CaseReport(
        case_id="x",
        task="t",
        replay_count=2,
        decision_denominator=2,
        trajectory_denominator=2,
        dar=1.0,
        tar=TARReport(seq=1.0, bag=1.0, set=1.0, strong=0.5),
        gap=0.0,
        eligibility=eligibility,
        unanimous_with_sequence_change=False,
        unanimous_with_path_change=True,
    )
    assert report.path_variation_kind.value == "argument_or_result"


def test_manifest_records_the_actual_package_version():
    suite = Suite.load("compliance-v1")
    manifest = build_manifest(
        suite,
        provider="fake",
        model="fake-v1",
        adapter="tests.version",
        request_parameters={"temperature": 0.0, "top_p": 1.0, "seed": 42},
        git_sha="f" * 40,
    )
    assert manifest.library_version == dfah.__version__


@pytest.mark.parametrize(
    "version",
    ["0.1.0", "1.0.0-alpha.1+build.7", "12.34.56+sha.abcdef"],
)
def test_suite_version_accepts_semver_2(version):
    base = Suite.load("compliance-v1")
    assert base.model_copy(update={"suite_version": version}).model_dump()
    Suite.model_validate({**base.model_dump(mode="json"), "suite_version": version})


@pytest.mark.parametrize("version", ["01.0.0", "1.00.0", "1.0.00", "1.0", "v1.0.0"])
def test_suite_version_rejects_non_semver(version):
    base = Suite.load("compliance-v1")
    with pytest.raises(ValidationError):
        Suite.model_validate({**base.model_dump(mode="json"), "suite_version": version})
