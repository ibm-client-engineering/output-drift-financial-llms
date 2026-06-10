"""L2 Hash Chain for tamper-evident agent execution logs.

Inspired by the AEGIS protocol (Li, 2026), Apache-2.0 license.
See https://github.com/crabsatellite/aegis-protocol

Implements a SHA-256 hash chain with three integrity invariants:
    I1: Strict sequence monotonicity (seq increments by 1)
    I2: Non-decreasing timestamps
    I3: Cryptographic linking (each event's prev_hash = prior event's hash)

Dependencies: hashlib (stdlib), bench.provenance.canonicalize.
"""

import hashlib
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .canonicalize import canonicalize


HASH_ALGORITHM = "sha256"


def _sha256(*parts: bytes) -> str:
    """Compute SHA-256 of concatenated byte parts, return hex digest."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def compute_genesis_hash(chain_id: str, agent_id: str, created_at: str) -> str:
    """Compute the genesis hash for a new chain.

    H_0 = SHA-256(encode(chain_id) || encode(agent_id) || encode(created_at))
    """
    return _sha256(
        chain_id.encode("utf-8"),
        agent_id.encode("utf-8"),
        created_at.encode("utf-8"),
    )


def compute_payload_hash(payload: Dict[str, Any]) -> str:
    """Compute SHA-256 of a canonicalized payload dict."""
    return _sha256(canonicalize(payload))


def compute_event_hash(
    seq: int,
    event_type: str,
    timestamp: str,
    payload_hash: str,
    prev_hash: str,
) -> str:
    """Compute the hash of a chain event.

    H_n = SHA-256(encode(seq, 8-byte BE) || encode(event_type) ||
                  encode(timestamp) || encode(payload_hash) || encode(prev_hash))
    """
    return _sha256(
        struct.pack(">Q", seq),
        event_type.encode("utf-8"),
        timestamp.encode("utf-8"),
        payload_hash.encode("utf-8"),
        prev_hash.encode("utf-8"),
    )


@dataclass
class ChainEvent:
    """A single event in the hash chain."""
    seq: int
    event_type: str
    timestamp: str
    payload_hash: str
    prev_hash: str
    event_hash: str


class Chain:
    """Stateful builder for an append-only hash chain.

    Usage:
        chain = Chain("chain-001", "agent-alice")
        chain.append("compliance.replay", {"decision": "escalate"})
        chain.append("compliance.replay", {"decision": "dismiss"})
        assert chain.verify()
        bundle_data = chain.to_dict()
    """

    def __init__(self, chain_id: str, agent_id: str):
        self.chain_id = chain_id
        self.agent_id = agent_id
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.genesis_hash = compute_genesis_hash(
            self.chain_id, self.agent_id, self.created_at
        )
        self.events: List[ChainEvent] = []

    @property
    def head(self) -> str:
        """Hash of the most recent event (or genesis hash if empty)."""
        if self.events:
            return self.events[-1].event_hash
        return self.genesis_hash

    @property
    def length(self) -> int:
        """Number of events in the chain."""
        return len(self.events)

    def append(
        self,
        event_type: str,
        payload: Dict[str, Any],
        timestamp: Optional[str] = None,
    ) -> ChainEvent:
        """Append a new event to the chain.

        Args:
            event_type: Event type string (e.g., "compliance.replay").
            payload: Event payload dict (only its hash is stored).
            timestamp: ISO 8601 timestamp. Defaults to current UTC time.

        Returns:
            The newly created ChainEvent.

        Raises:
            ValueError: If timestamp is earlier than the previous event (I2).
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        # I2: Non-decreasing timestamps
        if self.events and timestamp < self.events[-1].timestamp:
            raise ValueError(
                f"Timestamp {timestamp} is earlier than previous event "
                f"timestamp {self.events[-1].timestamp} (I2 violation)"
            )

        seq = len(self.events)
        prev_hash = self.head
        payload_hash = compute_payload_hash(payload)
        event_hash = compute_event_hash(
            seq, event_type, timestamp, payload_hash, prev_hash
        )

        event = ChainEvent(
            seq=seq,
            event_type=event_type,
            timestamp=timestamp,
            payload_hash=payload_hash,
            prev_hash=prev_hash,
            event_hash=event_hash,
        )
        self.events.append(event)
        return event

    def verify(self) -> bool:
        """Verify the entire chain's integrity.

        Checks I1 (monotonic seq), I2 (non-decreasing timestamps),
        I3 (prev_hash links), and recomputes all hashes.

        Returns:
            True if the chain is valid.
        """
        valid, error = verify_chain(self.to_dict())
        return valid

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the chain to a plain dict."""
        return {
            "chain_id": self.chain_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "genesis_hash": self.genesis_hash,
            "events": [
                {
                    "seq": e.seq,
                    "event_type": e.event_type,
                    "timestamp": e.timestamp,
                    "payload_hash": e.payload_hash,
                    "prev_hash": e.prev_hash,
                    "event_hash": e.event_hash,
                }
                for e in self.events
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chain":
        """Deserialize a chain from a plain dict."""
        chain = cls.__new__(cls)
        chain.chain_id = data["chain_id"]
        chain.agent_id = data["agent_id"]
        chain.created_at = data["created_at"]
        chain.genesis_hash = data["genesis_hash"]
        chain.events = [
            ChainEvent(
                seq=e["seq"],
                event_type=e["event_type"],
                timestamp=e["timestamp"],
                payload_hash=e["payload_hash"],
                prev_hash=e["prev_hash"],
                event_hash=e["event_hash"],
            )
            for e in data["events"]
        ]
        return chain


def verify_chain(chain_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Verify chain integrity from serialized form.

    Checks:
        I1: Monotonically increasing sequence numbers (0, 1, 2, ...)
        I2: Non-decreasing timestamps
        I3: Each event's prev_hash matches previous event's event_hash
        Hash recomputation: all event_hashes match expected values

    Args:
        chain_data: Serialized chain dict (from Chain.to_dict()).

    Returns:
        (valid, error_message) — error_message is None if valid.
    """
    try:
        genesis_hash = compute_genesis_hash(
            chain_data["chain_id"],
            chain_data["agent_id"],
            chain_data["created_at"],
        )
    except (KeyError, TypeError) as e:
        return False, f"Malformed chain data: missing field {e}"

    if genesis_hash != chain_data.get("genesis_hash"):
        return False, "Genesis hash mismatch"

    events = chain_data["events"]
    prev_hash = genesis_hash

    for i, event in enumerate(events):
        # I1: Monotonic sequence
        if event["seq"] != i:
            return False, f"I1 violation: expected seq {i}, got {event['seq']}"

        # I2: Non-decreasing timestamps
        if i > 0 and event["timestamp"] < events[i - 1]["timestamp"]:
            return False, (
                f"I2 violation at seq {i}: timestamp {event['timestamp']} "
                f"< previous {events[i - 1]['timestamp']}"
            )

        # I3: prev_hash linkage
        if event["prev_hash"] != prev_hash:
            return False, (
                f"I3 violation at seq {i}: prev_hash mismatch "
                f"(expected {prev_hash[:16]}..., got {event['prev_hash'][:16]}...)"
            )

        # Recompute event hash
        expected_hash = compute_event_hash(
            event["seq"],
            event["event_type"],
            event["timestamp"],
            event["payload_hash"],
            event["prev_hash"],
        )
        if expected_hash != event["event_hash"]:
            return False, f"Hash mismatch at seq {i}"

        prev_hash = event["event_hash"]

    return True, None
