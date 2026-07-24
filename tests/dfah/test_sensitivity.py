from __future__ import annotations

from dfah import ChannelState, ToolCall, Trajectory
from dfah.metrics import (
    adversarial_fallback_bound,
    case_resampling_interval,
    leave_one_case_out,
    sign_flip_permutation,
    subsample_replays,
)


def trajectory(*names: str) -> Trajectory:
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


def test_sensitivity_functions_are_seeded_and_bounded():
    interval = case_resampling_interval([0.0, 0.2, 0.4], n=100, seed=42)
    assert interval == case_resampling_interval([0.0, 0.2, 0.4], n=100, seed=42)
    assert interval.low <= interval.point <= interval.high
    permutation = sign_flip_permutation([0.1, 0.2, 0.3], n=1000, seed=42)
    assert 0 < permutation.p_value <= 1
    loo = leave_one_case_out([0.0, 0.5, 1.0])
    assert loo.minimum <= loo.full <= loo.maximum


def test_subsampling_preserves_shared_denominator():
    groups = [
        (
            ["a", "a", "a", "b"],
            [trajectory("x"), trajectory("x"), trajectory("y"), trajectory("x")],
        )
    ]
    result = subsample_replays(groups, k=3, draws=50, seed=7)
    assert result.k == 3 and result.draws == 50
    assert 0 <= result.dar_median <= 1
    assert 0 <= result.tar_median <= 1


def test_adversarial_fallback_bound_keeps_paths_and_denominator_fixed():
    groups = [
        (
            ["pass", "pass", "pass", "pass"],
            [trajectory("a"), trajectory("a"), trajectory("b"), trajectory("b")],
        )
    ]
    result = adversarial_fallback_bound(groups, [0.0, 0.25, 0.5])
    assert result.baseline_dar == 1.0
    assert result.baseline_tar == 0.5
    assert [point.tar for point in result.points] == [0.5, 0.5, 0.5]
    assert [point.episodes_flipped for point in result.points] == [0, 1, 2]
    assert result.points[1].dar < result.baseline_dar
    assert result.first_sign_change_fraction == 0.5
