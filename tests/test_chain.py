"""Tests for provenance hash chain.

Includes genesis creation, event append, roundtrip serialization,
and 6 tamper vectors that must all be detected.
"""

import copy
import pytest
from bench.provenance.chain import Chain, verify_chain, compute_genesis_hash


def _make_chain(n_events: int = 3) -> Chain:
    """Helper: create a chain with n events."""
    chain = Chain("test-chain", "test-agent")
    for i in range(n_events):
        chain.append(
            f"test.event_{i}",
            {"data": f"payload_{i}", "index": i},
            timestamp=f"2026-03-22T10:0{i}:00Z",
        )
    return chain


class TestChainBasic:

    def test_genesis(self):
        chain = Chain("chain-001", "agent-alice")
        assert chain.chain_id == "chain-001"
        assert chain.agent_id == "agent-alice"
        assert chain.genesis_hash  # non-empty
        assert chain.length == 0

    def test_genesis_hash_deterministic(self):
        h1 = compute_genesis_hash("c", "a", "2026-01-01T00:00:00Z")
        h2 = compute_genesis_hash("c", "a", "2026-01-01T00:00:00Z")
        assert h1 == h2

    def test_append(self):
        chain = _make_chain(3)
        assert chain.length == 3
        assert chain.events[0].seq == 0
        assert chain.events[1].seq == 1
        assert chain.events[2].seq == 2

    def test_head_updates(self):
        chain = Chain("c", "a")
        initial_head = chain.head
        chain.append("e", {"x": 1}, timestamp="2026-01-01T00:00:00Z")
        assert chain.head != initial_head

    def test_verify_valid_chain(self):
        chain = _make_chain(5)
        assert chain.verify()

    def test_verify_empty_chain(self):
        chain = Chain("c", "a")
        assert chain.verify()


class TestChainSerialization:

    def test_roundtrip(self):
        chain = _make_chain(3)
        data = chain.to_dict()
        restored = Chain.from_dict(data)
        assert restored.chain_id == chain.chain_id
        assert restored.agent_id == chain.agent_id
        assert restored.genesis_hash == chain.genesis_hash
        assert len(restored.events) == len(chain.events)
        for orig, rest in zip(chain.events, restored.events):
            assert orig.event_hash == rest.event_hash

    def test_serialized_verifies(self):
        chain = _make_chain(3)
        data = chain.to_dict()
        valid, error = verify_chain(data)
        assert valid, f"Verification failed: {error}"


class TestChainTamperDetection:
    """Six tamper vectors — all must be detected."""

    def test_tamper_1_modify_payload_hash(self):
        """Modify an event's payload_hash → hash mismatch."""
        chain = _make_chain(3)
        data = chain.to_dict()
        data["events"][1]["payload_hash"] = "0" * 64
        valid, error = verify_chain(data)
        assert not valid
        assert "mismatch" in error.lower() or "hash" in error.lower()

    def test_tamper_2_delete_event(self):
        """Delete an event → I1 sequence gap."""
        chain = _make_chain(3)
        data = chain.to_dict()
        del data["events"][1]  # Remove middle event
        valid, error = verify_chain(data)
        assert not valid
        assert "I1" in error or "seq" in error.lower()

    def test_tamper_3_reorder_events(self):
        """Swap two events → I3 prev_hash mismatch."""
        chain = _make_chain(3)
        data = chain.to_dict()
        data["events"][1], data["events"][2] = data["events"][2], data["events"][1]
        valid, error = verify_chain(data)
        assert not valid

    def test_tamper_4_backward_timestamp(self):
        """Set a timestamp earlier than previous → I2 violation."""
        chain = _make_chain(3)
        data = chain.to_dict()
        data["events"][2]["timestamp"] = "2000-01-01T00:00:00Z"
        valid, error = verify_chain(data)
        assert not valid
        assert "I2" in error or "timestamp" in error.lower()

    def test_tamper_5_modify_genesis_hash(self):
        """Modify genesis_hash → first event's prev_hash won't match."""
        chain = _make_chain(3)
        data = chain.to_dict()
        data["genesis_hash"] = "0" * 64
        valid, error = verify_chain(data)
        assert not valid

    def test_tamper_6_duplicate_seq(self):
        """Duplicate an event (same seq) → I1 violation."""
        chain = _make_chain(3)
        data = chain.to_dict()
        data["events"].append(copy.deepcopy(data["events"][2]))
        valid, error = verify_chain(data)
        assert not valid


class TestChainEdgeCases:

    def test_backward_timestamp_rejected(self):
        """Appending with backward timestamp raises ValueError."""
        chain = Chain("c", "a")
        chain.append("e1", {"x": 1}, timestamp="2026-03-22T10:00:00Z")
        with pytest.raises(ValueError, match="I2"):
            chain.append("e2", {"x": 2}, timestamp="2020-01-01T00:00:00Z")
