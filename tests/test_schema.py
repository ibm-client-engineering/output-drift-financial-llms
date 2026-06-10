"""Tests for DFAH-Bench schema and channel detection."""

import json
import tempfile
from pathlib import Path

import pytest

from bench.spec.schema import (
    Decision,
    DivergenceChannel,
    EvidenceContact,
    ReasoningTrace,
    ReplayEpisode,
    RunMetadata,
    ToolCall,
    available_channels,
    episodes_common_channels,
    load_episode,
    load_episodes,
)


class TestReplayEpisodeBasic:
    """Core episode construction and field access."""

    def test_minimal_episode(self):
        """Episode with only required fields."""
        ep = ReplayEpisode(
            case_id="TXN-001", benchmark="compliance",
            run_id=0, decision=Decision(label="escalate"),
        )
        assert ep.case_id == "TXN-001"
        assert ep.decision.label == "escalate"
        assert ep.tool_calls is None
        assert ep.evidence_contacts is None
        assert ep.reasoning_trace is None

    def test_full_episode(self):
        """Episode with all channels populated."""
        ep = ReplayEpisode(
            case_id="TXN-001", benchmark="compliance", run_id=0,
            decision=Decision(label="escalate", confidence=0.95),
            metadata=RunMetadata(model_name="test-model", temperature=0.0),
            tool_calls=[ToolCall(name="check_sanctions", output_hash="abc123")],
            evidence_contacts=[EvidenceContact(source_id="doc_1", contact_type="sanctions")],
            reasoning_trace=ReasoningTrace(raw_text="The entity is sanctioned."),
        )
        assert ep.metadata.model_name == "test-model"
        assert len(ep.tool_calls) == 1
        assert len(ep.evidence_contacts) == 1
        assert ep.reasoning_trace.raw_text == "The entity is sanctioned."

    def test_episode_is_frozen(self):
        ep = ReplayEpisode(
            case_id="TXN-001", benchmark="compliance",
            run_id=0, decision=Decision(label="escalate"),
        )
        with pytest.raises(AttributeError):
            ep.case_id = "changed"  # type: ignore


class TestChannelDetection:
    """Channel availability detection."""

    def test_no_channels(self):
        ep = ReplayEpisode(
            case_id="X", benchmark="compliance",
            run_id=0, decision=Decision(label="escalate"),
        )
        assert available_channels(ep) == set()

    def test_trajectory_channel(self):
        ep = ReplayEpisode(
            case_id="X", benchmark="compliance", run_id=0,
            decision=Decision(label="escalate"),
            tool_calls=[ToolCall(name="check_sanctions")],
        )
        channels = available_channels(ep)
        assert DivergenceChannel.TRAJECTORY in channels
        assert DivergenceChannel.EVIDENCE_CONTACT not in channels

    def test_evidence_channel(self):
        ep = ReplayEpisode(
            case_id="X", benchmark="compliance", run_id=0,
            decision=Decision(label="escalate"),
            evidence_contacts=[EvidenceContact(source_id="doc_1")],
        )
        assert DivergenceChannel.EVIDENCE_CONTACT in available_channels(ep)

    def test_rationale_channel(self):
        ep = ReplayEpisode(
            case_id="X", benchmark="compliance", run_id=0,
            decision=Decision(label="escalate"),
            reasoning_trace=ReasoningTrace(raw_text="This is a detailed explanation of the decision."),
        )
        assert DivergenceChannel.RATIONALE in available_channels(ep)

    def test_rationale_too_short_not_detected(self):
        """Short reasoning text (<=20 chars) is not considered a valid channel."""
        ep = ReplayEpisode(
            case_id="X", benchmark="compliance", run_id=0,
            decision=Decision(label="escalate"),
            reasoning_trace=ReasoningTrace(raw_text="short"),
        )
        assert DivergenceChannel.RATIONALE not in available_channels(ep)

    def test_empty_tool_calls_not_detected(self):
        ep = ReplayEpisode(
            case_id="X", benchmark="compliance", run_id=0,
            decision=Decision(label="escalate"),
            tool_calls=[],
        )
        assert DivergenceChannel.TRAJECTORY not in available_channels(ep)

    def test_common_channels(self):
        ep1 = ReplayEpisode(
            case_id="X", benchmark="c", run_id=0,
            decision=Decision(label="a"),
            tool_calls=[ToolCall(name="t1")],
            evidence_contacts=[EvidenceContact(source_id="d1")],
        )
        ep2 = ReplayEpisode(
            case_id="X", benchmark="c", run_id=1,
            decision=Decision(label="a"),
            tool_calls=[ToolCall(name="t2")],
        )
        common = episodes_common_channels([ep1, ep2])
        assert DivergenceChannel.TRAJECTORY in common
        assert DivergenceChannel.EVIDENCE_CONTACT not in common  # ep2 lacks it


class TestLoadEpisode:
    """Loading episodes from JSON files."""

    def test_load_from_json(self):
        """Load a synthetic run log JSON and verify canonical fields."""
        data = {
            "model": "test-model",
            "benchmark": "compliance",
            "case_id": "TXN-2025-001",
            "run_id": 0,
            "seed": 42,
            "temperature": 0.0,
            "timestamp": "2026-03-04T21:49:12.551188",
            "tool_sequence": ["check_sanctions", "get_customer_profile"],
            "tool_output_hashes": ["abc123", "def456"],
            "decision_output": "escalate",
            "deterministic": True,
            "faithfulness_score": 0.85,
            "runtime_seconds": 7.645,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="case_TXN-001_run_0",
            delete=False,
        ) as f:
            json.dump(data, f)
            path = Path(f.name)

        try:
            ep = load_episode(path)
            assert ep.case_id == "TXN-2025-001"
            assert ep.benchmark == "compliance"
            assert ep.decision.label == "escalate"
            assert ep.metadata.model_name == "test-model"
            assert ep.metadata.temperature == 0.0
            assert ep.metadata.seed == 42
            assert len(ep.tool_calls) == 2
            assert ep.tool_calls[0].name == "check_sanctions"
            assert ep.tool_calls[0].output_hash == "abc123"
            assert ep.reasoning_trace is None
        finally:
            path.unlink()

    def test_load_empty_tool_sequence(self):
        """Run log with empty tool_sequence → tool_calls is None."""
        data = {
            "model": "mistral_7b",
            "benchmark": "compliance",
            "case_id": "TXN-001",
            "run_id": 0,
            "tool_sequence": [],
            "tool_output_hashes": [],
            "decision_output": "investigate",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="case_TXN-001_run_0",
            delete=False,
        ) as f:
            json.dump(data, f)
            path = Path(f.name)

        try:
            ep = load_episode(path)
            assert ep.tool_calls is None
            assert DivergenceChannel.TRAJECTORY not in available_channels(ep)
        finally:
            path.unlink()
