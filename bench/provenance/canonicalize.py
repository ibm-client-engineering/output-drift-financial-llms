"""Deterministic JSON serialization for hash chain payloads.

Produces canonical byte representations suitable for SHA-256 hashing
and Ed25519 signing. Uses sorted keys and compact separators.

Inspired by RFC 8785 (JSON Canonicalization Scheme). Covers sorted keys
and compact serialization. Does not implement ES2019 number formatting
edge cases (subnormals, -0). See test suite for covered behaviors.

Dependencies: stdlib only.
"""

import json
from typing import Any


def canonicalize(obj: Any) -> bytes:
    """Serialize a Python object to deterministic canonical JSON bytes.

    - Object keys sorted lexicographically at every nesting level
    - No whitespace between tokens
    - UTF-8 encoded
    - Deterministic: same input always produces same output

    Args:
        obj: Any JSON-serializable Python object.

    Returns:
        Canonical UTF-8 bytes.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def canonicalize_str(obj: Any) -> str:
    """Serialize to canonical JSON string (convenience wrapper)."""
    return canonicalize(obj).decode("utf-8")
