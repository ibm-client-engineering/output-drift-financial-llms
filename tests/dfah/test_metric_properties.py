from __future__ import annotations

import pytest

from dfah import ChannelState, ToolCall, Trajectory
from dfah.metrics import dar, first_anchored_agreement, modal_agreement, tar
from dfah.parse import parse_decision

hypothesis = pytest.importorskip("hypothesis")
example = hypothesis.example
given = hypothesis.given
st = pytest.importorskip("hypothesis.strategies")

LABELS = st.sampled_from(["pass", "review", "reject"])
TOOL_NAMES = st.sampled_from(["profile", "sanctions", "limit", "ledger"])


def _trajectory(names: list[str]) -> Trajectory:
    calls = tuple(
        ToolCall(
            name=name,
            output_hash="0" * 64,
            result_state=ChannelState.OBSERVED_NONEMPTY,
        )
        for name in names
    )
    return Trajectory(
        state=ChannelState.OBSERVED_NONEMPTY if calls else ChannelState.OBSERVED_EMPTY,
        tool_calls=calls,
    )


@given(st.lists(LABELS, min_size=1, max_size=20))
def test_modal_agreement_bounds_and_dominates_first_anchor(values: list[str]):
    modal = modal_agreement(values).value
    assert 1 / len(values) <= dar(values) <= 1.0
    assert modal >= first_anchored_agreement(values)


@given(
    st.lists(
        st.lists(TOOL_NAMES, min_size=0, max_size=5),
        min_size=1,
        max_size=12,
    )
)
def test_trajectory_coarsening_is_monotone(raw_paths: list[list[str]]):
    paths = [_trajectory(names) for names in raw_paths]
    assert tar(paths, "set") >= tar(paths, "bag") >= tar(paths, "seq")
    assert tar(paths, "strong") <= tar(paths, "seq")


@given(st.text(max_size=100).filter(lambda value: "DECISION:" not in value.upper()))
@example("pass")
@example("I would probably review this.")
@example("DECISION = REJECT")
def test_malformed_decisions_never_gain_a_fallback_label(text: str):
    decision, provenance = parse_decision(text, ("pass", "review", "reject"))
    assert decision is None
    assert provenance.fallback
    assert provenance.strategy == "none"
    assert not provenance.accepted


@given(LABELS, LABELS)
def test_duplicate_decision_markers_fail_closed(first: str, second: str):
    decision, provenance = parse_decision(
        f"DECISION: {first}\nDECISION: {second}",
        ("pass", "review", "reject"),
    )
    assert decision is None
    assert provenance.fallback
