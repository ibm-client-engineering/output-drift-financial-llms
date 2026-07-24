"""Required-channel eligibility, kept separate from operational status."""

from __future__ import annotations

from collections.abc import Sequence

from ..models import ChannelState, Eligibility, Episode, EpisodeStatus


def evaluate_episode(episode: Episode, *, require_evidence: bool = False) -> Eligibility:
    """Evaluate one episode without converting missing data to empty activity."""

    decision_state = (
        ChannelState.OBSERVED_NONEMPTY
        if episode.status is EpisodeStatus.VALID and episode.decision is not None
        else ChannelState.UNAVAILABLE
    )
    trajectory_state = episode.trajectory.state
    evidence_state = ChannelState.UNAVAILABLE
    reasons: list[str] = []
    if episode.status is not EpisodeStatus.VALID:
        reasons.append(f"terminal_status_{episode.status.value}")
    if not decision_state.observed:
        reasons.append("decision_missing_or_invalid")
    if not trajectory_state.observed:
        reasons.append("trajectory_missing_or_malformed")
    if require_evidence and not evidence_state.observed:
        reasons.append("evidence_missing_or_malformed")
    return Eligibility(
        eligible=not reasons,
        decision=decision_state,
        trajectory=trajectory_state,
        evidence=evidence_state,
        reasons=tuple(reasons),
    )


def evaluate_group(
    episodes: Sequence[Episode],
    *,
    required_replays: int,
    require_evidence: bool = False,
) -> Eligibility:
    """Require exact replay count, one manifest, and all required channels."""

    reasons: list[str] = []
    if len(episodes) != required_replays:
        reasons.append(f"expected_{required_replays}_replays_observed_{len(episodes)}")
    if len({episode.manifest_hash for episode in episodes}) > 1:
        reasons.append("manifest_mismatch")
    request_hashes = {
        episode.wire_request.payload_hash
        for episode in episodes
        if episode.wire_request is not None
    }
    if len(request_hashes) > 1:
        reasons.append("wire_payload_mismatch")
    endpoints = {
        episode.wire_request.endpoint
        for episode in episodes
        if episode.wire_request is not None
    }
    if len(endpoints) > 1:
        reasons.append("wire_endpoint_mismatch")
    per_episode = [
        evaluate_episode(episode, require_evidence=require_evidence) for episode in episodes
    ]
    for index, result in enumerate(per_episode):
        reasons.extend(f"replay_{index}:{reason}" for reason in result.reasons)
    decision = (
        ChannelState.OBSERVED_NONEMPTY
        if episodes and all(result.decision.observed for result in per_episode)
        else ChannelState.UNAVAILABLE
    )
    if episodes and all(result.trajectory.observed for result in per_episode):
        trajectory = (
            ChannelState.OBSERVED_NONEMPTY
            if any(episode.trajectory.tool_calls for episode in episodes)
            else ChannelState.OBSERVED_EMPTY
        )
    else:
        trajectory = ChannelState.UNAVAILABLE
    evidence = ChannelState.UNAVAILABLE
    return Eligibility(
        eligible=not reasons,
        decision=decision,
        trajectory=trajectory,
        evidence=evidence,
        reasons=tuple(reasons),
    )
